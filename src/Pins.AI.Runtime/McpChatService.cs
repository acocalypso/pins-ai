using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Pins.AI.Runtime;

public sealed class McpChatService(LlamaRuntime runtime, string mcpToken,
    Func<string, RigContext> hostIdentity, Func<HttpMessageHandler?>? transport = null) : IAsyncDisposable {
    private sealed class Session(McpClient client) : IAsyncDisposable {
        public McpClient Client { get; } = client;
        public ToolGateway Gateway { get; set; } = null!;
        public ApplicationOperatorAuthority Authority { get; set; } = null!;
        public ConcurrentDictionary<string, ToolPreview> Previews { get; } = new();
        public List<JsonArray> History { get; } = [];
        public string[] ClarificationTools { get; set; } = [];
        public SemaphoreSlim Operation { get; } = new(1, 1);
        public DateTimeOffset Expires { get; } = DateTimeOffset.UtcNow.AddMinutes(15);
        public async ValueTask DisposeAsync() { await Client.DisposeAsync(); }
    }
    private readonly ConcurrentDictionary<string, Session> sessions = new();
    private readonly SemaphoreSlim registrations = new(1, 1);
    private string? activeInference;
    private static readonly HashSet<string> Reads = ["get_status", "get_recent_images", "get_sequence", "list_saved_sequences", "list_target_templates", "get_job", "list_jobs"];

    public async Task<JsonObject> OpenAsync(CancellationToken ct) {
        await registrations.WaitAsync(ct);
        try {
            foreach (var expired in sessions.Where(s => s.Value.Expires <= DateTimeOffset.UtcNow).Select(s => s.Key).ToArray()) await CloseAsync(expired);
            if (sessions.Count >= 8) throw new AiBusyException();
            var id = Convert.ToHexString(RandomNumberGenerator.GetBytes(32));
            var session = new Session(new McpClient(new Uri("http://127.0.0.1:4782/mcp"), mcpToken, transport?.Invoke()));
            try {
                var catalog = await session.Client.DiscoverAsync(ct);
                session.Authority = new ApplicationOperatorAuthority(
                    (supplied, _) => Task.FromResult(supplied == id && sessions.TryGetValue(id, out var active) &&
                        ReferenceEquals(active, session) && active.Expires > DateTimeOffset.UtcNow ? id : null),
                    async (_, chat, token) => await CaptureAsync(session, chat, token));
                session.Gateway = new ToolGateway(catalog, authority: session.Authority);
                sessions[id] = session;
                RigContext identity;
                try { identity = await CaptureAsync(session, id, ct); }
                catch (ArgumentException) { identity = hostIdentity(id) with { StateRevision = null }; }
                // Unknown live state exposes only queries; previews always refresh it.
                var exposed = session.Gateway.ExposedTools(identity, proposals: true);
                return new JsonObject { ["sessionId"] = id, ["expires"] = session.Expires.ToString("O"),
                    ["tools"] = exposed["tools"]!.DeepClone(), ["status"] = JsonSerializer.SerializeToNode(runtime.Status, AiConfiguration.Json) };
            }
            catch { sessions.TryRemove(id, out _); await session.DisposeAsync(); throw; }
        }
        finally { registrations.Release(); }
    }

    private async Task<RigContext> CaptureAsync(Session session, string id, CancellationToken ct) {
        var host = hostIdentity(id);
        var status = await session.Gateway.QueryStateAsync(session.Client, "get_status", new(), ct);
        var sequence = await session.Gateway.QueryStateAsync(session.Client, "get_sequence", new() { ["max_depth"] = 1 }, ct);
        var context = McpRigStateReader.Parse(status, sequence, host, TimeProvider.System);
        // Bind approvals to relevant live state without temperature/poll timestamps.
        var states = new JsonArray();
        foreach (var device in status["devices"]!.AsArray()) {
            var state = new JsonObject { ["device"] = device!["device"]!.DeepClone(), ["connected"] = device["connected"]!.DeepClone() };
            foreach (var key in new[] { "isExposing", "slewing", "parked", "tracking", "moving", "state" })
                if (device["properties"]?[key] is { } value) state[key] = value.DeepClone();
            states.Add(state);
        }
        var revision = host.StateRevision + states.ToJsonString() + status["jobs"]?.ToJsonString();
        return context with { StateRevision = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(revision))) };
    }

    public async Task<JsonObject> HandleAsync(string id, string action, JsonObject body, CancellationToken ct) {
        if (!sessions.TryGetValue(id, out var session) || session.Expires <= DateTimeOffset.UtcNow)
            throw new ArgumentException("Operator session expired; reconnect.");
        if (!await session.Operation.WaitAsync(0, ct)) throw new AiBusyException();
        try {
            if (action == "clear") {
                ExactFields(body);
                session.History.Clear(); session.ClarificationTools = []; session.Previews.Clear(); session.Gateway.Clear(id);
                return new JsonObject { ["kind"] = "cleared" };
            }
            if (action == "chat") {
                if (body.Any(p => p.Key is not ("messages" or "locale" or "toolNames"))) throw new ArgumentException("Unexpected chat field.");
                var request = new ChatRequest(body["messages"]?.Deserialize<ChatMessage[]>(AiConfiguration.Json) ?? [], body["locale"]?.GetValue<string>());
                ChatPolicy.Validate(request);
                RigContext identity;
                try { identity = await CaptureAsync(session, id, ct); }
                catch (ArgumentException) { identity = hostIdentity(id) with { StateRevision = null }; }
                var catalog = session.Gateway.ExposedTools(identity, proposals: true)["tools"]!.AsArray();
                var selected = body["toolNames"]?.AsArray().Select(n => n!.GetValue<string>()).ToArray()
                    ?? ChatToolRouter.Select(request.Messages[^1].Content, catalog, session.ClarificationTools);
                if (selected.Length > 4 || selected.Distinct().Count() != selected.Length) throw new ArgumentException("Supply at most four distinct tools.");
                var tools = new JsonArray();
                foreach (var name in selected) {
                    var tool = catalog.FirstOrDefault(t => t?["name"]?.GetValue<string>() == name) ?? throw new ArgumentException("Tool is not exposed by this gateway.");
                    tools.Add(new JsonObject { ["type"] = "function", ["function"] = new JsonObject {
                        ["name"] = name, ["description"] = name == "add_target" ? ToolPromptSchema.TargetDescription : tool["description"]!.GetValue<string>()[..Math.Min(180, tool["description"]!.GetValue<string>().Length)],
                        ["parameters"] = ToolPromptSchema.Compact(tool["inputSchema"]!)
                    }});
                }
                // Store trusted tool feedback on the server. Request JSON cannot inject
                // tool results, system messages, approval state or native tool history.
                var turn = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = request.Messages[^1].Content });
                var history = new JsonArray();
                foreach (var previous in session.History.TakeLast(session.ClarificationTools.Length > 0 ? 4 : 2)) foreach (var item in previous) history.Add(item!.DeepClone());
                history.Add(turn[0]!.DeepClone());
                if (Interlocked.CompareExchange(ref activeInference, id, null) is not null) throw new AiBusyException();
                JsonObject generated;
                try { generated = await runtime.ProposeAsync(request, tools, ct, history); }
                finally { Interlocked.CompareExchange(ref activeInference, null, id); }
                var proposal = ParseProposal(generated["rawText"]!.GetValue<string>());
                if (proposal is null) {
                    session.ClarificationTools = generated["rawText"]!.GetValue<string>().IndexOfAny(['?', '？']) >= 0 ? selected : [];
                    turn.Add(new JsonObject { ["role"] = "assistant", ["content"] = generated["rawText"]!.GetValue<string>() });
                    Remember(session, turn); generated["kind"] = "text"; generated["verifiedAction"] = false; return generated;
                }
                var toolName = proposal["name"]!.GetValue<string>();
                if (!selected.Contains(toolName)) throw new ArgumentException("Model proposed a tool outside its retrieved context. Rephrase the request with the operation and device.");
                session.ClarificationTools = [];
                Remember(session, turn);
                var response = await ProposeAsync(session, id, toolName, proposal["arguments"]!.AsObject(), ct);
                response["generation"] = generated;
                return response;
            }
            if (action == "tool") {
                ExactFields(body, "name", "arguments");
                return await ProposeAsync(session, id, body["name"]!.GetValue<string>(), body["arguments"]!.AsObject(), ct);
            }
            if (action == "approve") {
                ExactFields(body, "requestId", "name", "arguments");
                var requestId = body["requestId"]!.GetValue<string>();
                if (!session.Previews.TryRemove(requestId, out var preview)) throw new ArgumentException("Unknown or consumed approval.");
                await session.Authority.DecideAsync(id, preview, OperatorDecision.Approved,
                    body["name"]!.GetValue<string>(), body["arguments"]!.AsObject(), ct: ct);
                var result = await session.Gateway.ExecuteApprovedAsync(session.Client, requestId, ct);
                RememberResult(session, preview.Tool, preview.Arguments, result);
                return Result(preview.Tool, result);
            }
            if (action == "reject") {
                ExactFields(body, "requestId");
                var requestId = body["requestId"]!.GetValue<string>();
                if (session.Previews.TryRemove(requestId, out var preview))
                    await session.Authority.DecideAsync(id, preview, OperatorDecision.Denied, preview.Tool, preview.Arguments, ct: ct);
                return new JsonObject { ["kind"] = "rejected" };
            }
            throw new ArgumentException("Unknown gateway action.");
        }
        finally { session.Operation.Release(); }
    }

    private async Task<JsonObject> ProposeAsync(Session session, string id, string name, JsonObject args, CancellationToken ct) {
        session.Gateway.Validate(name, args);
        if (Reads.Contains(name)) {
            var result = await session.Gateway.InvokeAsync(session.Client, name, args, hostIdentity(id), ct: ct);
            RememberResult(session, name, args, result);
            return Result(name, result);
        }
        var preview = await session.Gateway.PreviewAsync(name, args, hostIdentity(id).RigId, id, ct);
        foreach (var expired in session.Previews.Where(p => p.Value.Expires <= DateTimeOffset.UtcNow).Select(p => p.Key).ToArray()) session.Previews.TryRemove(expired, out _);
        session.Previews[preview.RequestId] = preview;
        return new JsonObject { ["kind"] = "approval", ["preview"] = JsonSerializer.SerializeToNode(preview, AiConfiguration.Json) };
    }
    private static JsonObject Result(string name, JsonObject result) => new() {
        ["kind"] = "result", ["name"] = name, ["result"] = result, ["verifiedAction"] = result["completionConfirmed"]?.DeepClone() ?? JsonValue.Create(false)
    };
    private static void Remember(Session session, JsonArray turn) {
        session.History.Add(turn);
        while (session.History.Count > 4) session.History.RemoveAt(0);
    }
    private static void RememberResult(Session session, string name, JsonObject args, JsonObject result) {
        session.ClarificationTools = [];
        var feedback = result;
        if (name == "get_status" && result["devices"] is JsonArray devices) {
            var summaries = new JsonArray();
            foreach (var device in devices) {
                var copy = new JsonObject { ["device"] = device!["device"]!.DeepClone(), ["connected"] = device["connected"]!.DeepClone() };
                if (device["name"] is { } display) copy["name"] = display.DeepClone();
                summaries.Add(copy);
            }
            feedback = new JsonObject { ["summaryOf"] = name, ["otherFieldsOmitted"] = true,
                ["pinsResponding"] = true, ["sequence"] = result["sequence"]?.DeepClone(), ["devices"] = summaries };
        }
        var content = feedback.ToJsonString();
        if (content.Length > 3000) content = new JsonObject { ["summaryOf"] = name, ["otherFieldsOmitted"] = true,
            ["isError"] = result["isError"]?.DeepClone(), ["job"] = result["job"]?.DeepClone(),
            ["completionConfirmed"] = result["completionConfirmed"]?.DeepClone() }.ToJsonString();
        Remember(session, new JsonArray(new JsonObject { ["role"] = "assistant", ["content"] = "",
            ["tool_calls"] = new JsonArray(new JsonObject { ["type"] = "function", ["function"] = new JsonObject {
                ["name"] = name, ["arguments"] = args.DeepClone() } }) },
            new JsonObject { ["role"] = "tool", ["name"] = name, ["content"] = content }));
    }
    private static void ExactFields(JsonObject body, params string[] fields) {
        if (body.Count != fields.Length || fields.Any(f => !body.ContainsKey(f))) throw new ArgumentException("Unexpected or missing request fields.");
    }
    public static JsonObject? ParseProposal(string raw) {
        var text = raw.Trim();
        if (!text.Contains("<tool_call>", StringComparison.Ordinal)) {
            if (text.Contains("</tool_call>", StringComparison.Ordinal)) throw new ArgumentException("Malformed model tool call.");
            return null;
        }
        if (!text.StartsWith("<tool_call>", StringComparison.Ordinal) || !text.EndsWith("</tool_call>", StringComparison.Ordinal) ||
            text.IndexOf("<tool_call>", 11, StringComparison.Ordinal) >= 0) throw new ArgumentException("Expected exactly one complete native tool call.");
        var proposal = JsonNode.Parse(text[11..^12], documentOptions: new() { MaxDepth = 16 })?.AsObject() ?? throw new ArgumentException("Invalid model tool call.");
        ExactFields(proposal, "name", "arguments");
        _ = proposal["name"]!.GetValue<string>(); _ = proposal["arguments"]!.AsObject();
        return proposal;
    }
    public async Task CloseAsync(string id) {
        if (activeInference == id) await runtime.CancelAsync();
        if (sessions.TryRemove(id, out var session)) {
            await session.Operation.WaitAsync();
            try { session.Gateway.Clear(id); await session.DisposeAsync(); }
            finally { session.Operation.Release(); }
        }
    }
    public async Task CancelAsync(string id) {
        if (!sessions.TryGetValue(id, out var session) || session.Expires <= DateTimeOffset.UtcNow) throw new ArgumentException("Operator session expired.");
        session.Previews.Clear(); session.Gateway.Clear(id);
        if (activeInference == id) await runtime.CancelAsync();
    }
    public async ValueTask DisposeAsync() {
        foreach (var id in sessions.Keys) await CloseAsync(id);
    }
}
