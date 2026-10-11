using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Pins.AI.Runtime;

public sealed record RigContext(string RigId, string ChatId, bool SequenceRunning, DateTimeOffset ObservedAt,
    IReadOnlyDictionary<int, (double Minimum, double Maximum)> SwitchRanges,
    IReadOnlyDictionary<int, double>? SwitchSteps = null,
    string? StateRevision = null, IReadOnlyDictionary<string, string>? DeviceIds = null,
    IReadOnlyDictionary<int, SwitchCapability>? SwitchCapabilities = null);
public sealed record ToolPreview(string RequestId, string Tool, JsonObject Arguments, string RigId, string ChatId, DateTimeOffset Expires);

// No caller/model-provided system string can authorize this gateway.
public sealed class ToolGateway {
    private static readonly HashSet<string> Reads = ["get_status", "get_recent_images", "get_sequence", "list_saved_sequences", "list_target_templates", "get_job", "list_jobs"];
    private static readonly HashSet<string> Allowed = [.. Reads, "stop_all", "cancel_job", "stop_sequence", "stop_guiding", "connect_device", "disconnect_device", "slew_mount", "park_mount", "unpark_mount", "home_mount", "set_tracking", "capture_image", "cool_camera", "warm_camera", "move_focuser", "run_autofocus", "change_filter", "start_guiding", "set_switch", "load_sequence", "add_target", "clear_targets", "save_sequence", "start_sequence", "reset_sequence"];
    private readonly Dictionary<string, JsonObject> catalog;
    private readonly Dictionary<string, (ToolPreview Preview, string Digest, string State)> pending = [];
    private readonly IOperatorAuthority? authority;
    private readonly TimeProvider clock;
    private readonly SemaphoreSlim calls = new(1, 1);
    public ToolGateway(JsonObject toolsList, TimeProvider? clock = null, IOperatorAuthority? authority = null) {
        this.clock = clock ?? TimeProvider.System;
        this.authority = authority;
        catalog = [];
        foreach (var item in toolsList["tools"]?.AsArray() ?? throw new ArgumentException("Missing tools/list")) {
            var tool = item!.AsObject();
            var name = tool["name"]!.GetValue<string>();
            if (Reads.Contains(name) && tool["annotations"]?["readOnlyHint"]?.GetValue<bool>() != true)
                throw new ArgumentException("Known query changed its read-only contract; re-audit before connecting.");
            if (Allowed.Contains(name)) catalog.Add(name, (JsonObject)tool.DeepClone());
        }
    }

    public void Validate(string name, JsonObject arguments) {
        arguments = JsonNode.Parse(arguments.ToJsonString())!.AsObject();
        if (!catalog.TryGetValue(name, out var tool)) throw new ArgumentException("Unknown or unexposed tool.");
        Schema(arguments, tool["inputSchema"]!.AsObject());
        foreach (var field in new[] { "force", "skip_validation", "overwrite" })
            if (arguments[field]?.GetValue<bool>() == true) throw new ArgumentException("Safety overrides are disabled.");
        if (arguments["wait_seconds"] is { } wait && wait.GetValue<double>() > 5) throw new ArgumentException("AI tool wait is limited to five seconds.");
        if (name == "change_filter" && arguments.ContainsKey("name") == arguments.ContainsKey("position")) throw new ArgumentException("Choose exactly one filter selector.");
        if (name == "add_target") {
            if (arguments["dither_every"] is { } dither && dither.GetValue<double>() > 32)
                throw new ArgumentException("NINA exposure-plan adapter limits dither_every to 32.");
            if (arguments.ContainsKey("template") == arguments.ContainsKey("exposures")) throw new ArgumentException("Choose exactly one target plan.");
            if (arguments.ContainsKey("template") && new[] { "slew_and_center", "start_guiding", "autofocus_at_start", "dither_every" }.Any(arguments.ContainsKey))
                throw new ArgumentException("Template forbids exposure-plan options.");
        }
        foreach (var field in new[] { "ra", "dec" })
            if (arguments[field] is JsonValue value && !value.TryGetValue<double>(out _)) throw new ArgumentException("AI requires verified numeric J2000 coordinates.");
        if (name == "save_sequence") {
            var file = arguments["name"]!.GetValue<string>().Trim();
            if (file.EndsWith(".json", StringComparison.OrdinalIgnoreCase)) file = file[..^5].TrimEnd();
            if (file.Length is < 1 or > 100 || file.StartsWith('.') || file.EndsWith('.') || file.Any(c => !char.IsLetterOrDigit(c) && !" -_.".Contains(c)))
                throw new ArgumentException("Unsafe sequence filename.");
            if (file.EndsWith(".template", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Sequence names ending in .template are reserved.");
        }
    }

    public ToolPreview Preview(string name, JsonObject arguments, RigContext context) {
        arguments = EffectiveArguments(name, arguments);
        Validate(name, arguments); ValidateContext(name, arguments, context);
        var preview = new ToolPreview(Convert.ToHexString(RandomNumberGenerator.GetBytes(24)), name,
            (JsonObject)arguments.DeepClone(), context.RigId, context.ChatId, clock.GetUtcNow().AddMinutes(1));
        lock (pending) {
            foreach (var expired in pending.Where(p => p.Value.Preview.Expires <= clock.GetUtcNow()).Select(p => p.Key).ToArray()) pending.Remove(expired);
            if (pending.Count >= 32) throw new ArgumentException("Too many pending confirmations.");
            pending.Add(preview.RequestId, (preview with { Arguments = (JsonObject)preview.Arguments.DeepClone() }, Digest(arguments), StateDigest(context)));
        }
        return preview;
    }

    // Kept for source compatibility, but text/IDs alone cannot grant authority.
    public void Confirm(string requestId, string rigId, string chatId) {
        throw new ArgumentException("Confirmation requires the trusted application authority, not supplied strings.");
    }

    public async Task<ToolPreview> PreviewAsync(string name, JsonObject arguments, string rigId, string chatId, CancellationToken ct = default) {
        if (authority is null) throw new ArgumentException("Trusted application authority is unavailable.");
        var context = await authority.RefreshAsync(rigId, chatId, ct);
        if (context.RigId != rigId || context.ChatId != chatId) throw new ArgumentException("Application context identity changed.");
        return Preview(name, arguments, context);
    }

    // Read-only revalidation while the mutation semaphore is held. The MCP
    // client serializes transport without recursively acquiring this gateway.
    internal async Task<JsonObject> QueryStateAsync(McpClient client, string name, JsonObject arguments, CancellationToken ct) {
        if (name is not ("get_status" or "get_sequence")) throw new ArgumentException("Only state queries are allowed here.");
        Validate(name, arguments);
        return Normalize(await client.CallAsync(name, (JsonObject)arguments.DeepClone(), ct));
    }

    public async Task<JsonObject> ExecuteApprovedAsync(McpClient client, string requestId, CancellationToken ct = default) {
        if (authority is null) throw new ArgumentException("Trusted application authority is unavailable.");
        ToolPreview preview;
        lock (pending) {
            if (!pending.TryGetValue(requestId, out var item)) throw new ArgumentException("Unknown or consumed preview.");
            preview = item.Preview with { Arguments = (JsonObject)item.Preview.Arguments.DeepClone() };
        }
        var context = await authority.RefreshAsync(preview.RigId, preview.ChatId, ct);
        return await InvokeAsync(client, preview.Tool, preview.Arguments, context, requestId, ct);
    }

    // Safe proposal catalog: exposure is not authorization. No bypass fields.
    public JsonObject ExposedTools(RigContext context, bool proposals = false) {
        var tools = new JsonArray();
        var age = clock.GetUtcNow() - context.ObservedAt;
        var canPropose = proposals && authority is not null && age >= TimeSpan.Zero && age <= TimeSpan.FromSeconds(10);
        try { _ = StateDigest(context); } catch (ArgumentException) { canPropose = false; }
        foreach (var (name, tool) in catalog) {
            if (!Reads.Contains(name) && (!canPropose ||
                context.SequenceRunning && name is not ("stop_all" or "stop_sequence" or "cancel_job") ||
                name == "set_switch" && context.SwitchCapabilities?.Values.Any(c => c.Writable) != true)) continue;
            var copy = (JsonObject)tool.DeepClone();
            var schema = copy["inputSchema"]!.AsObject();
            foreach (var field in new[] { "force", "skip_validation", "overwrite" }) schema["properties"]!.AsObject().Remove(field);
            if (schema["properties"]?["wait_seconds"] is JsonObject wait) { wait["maximum"] = 5; wait["default"] = 0; }
            foreach (var coordinate in new[] { "ra", "dec" }) {
                if (schema["properties"]?[coordinate] is JsonObject value && value["anyOf"] is JsonArray variants) {
                    var numeric = variants.First(v => v?["type"]?.GetValue<string>() == "number")!;
                    schema["properties"]![coordinate] = numeric.DeepClone();
                }
            }
            tools.Add(copy);
        }
        return new() { ["tools"] = tools };
    }

    public void Clear(string chatId) {
        lock (pending) foreach (var key in pending.Where(p => p.Value.Preview.ChatId == chatId).Select(p => p.Key).ToArray()) pending.Remove(key);
    }

    public async Task<JsonObject> InvokeAsync(McpClient client, string name, JsonObject arguments, RigContext context,
        string? approvalId = null, CancellationToken ct = default) {
        arguments = EffectiveArguments(name, arguments);
        if (!await calls.WaitAsync(0, ct)) throw new AiBusyException();
        try {
            Validate(name, arguments); ValidateContext(name, arguments, context);
            if (!Reads.Contains(name)) {
                if (authority is null) throw new ArgumentException("Trusted application authority is unavailable; mutation disabled.");
                (ToolPreview Preview, string Digest, string State) item;
                lock (pending) {
                    if (approvalId is null || !pending.Remove(approvalId, out item) ||
                        item.Preview.Expires <= clock.GetUtcNow() || item.Preview.Tool != name || item.Preview.RigId != context.RigId ||
                        item.Preview.ChatId != context.ChatId || item.Digest != Digest(arguments))
                        throw new ArgumentException("Exact, unexpired operator confirmation is required.");
                }
                var liveCatalog = await client.DiscoverAsync(ct);
                var liveTool = liveCatalog["tools"]?.AsArray().FirstOrDefault(t => t?["name"]?.GetValue<string>() == name);
                if (liveTool is null || Digest(liveTool) != Digest(catalog[name]))
                    throw new ArgumentException("Published tool contract changed; re-audit before execution.");
                // Re-read state after approval and immediately before the MCP call.
                context = await authority.RefreshAsync(context.RigId, context.ChatId, ct);
                ValidateContext(name, arguments, context);
                if (context.RigId != item.Preview.RigId || context.ChatId != item.Preview.ChatId ||
                    item.State != StateDigest(context) || item.Preview.Expires <= clock.GetUtcNow())
                    throw new ArgumentException("Device, session or state changed; obtain a new preview and approval.");
                // Authentication is refreshed last, after potentially slow MCP
                // queries, so revocation during those queries cannot authorize a call.
                var approval = await authority.GetApprovalAsync(approvalId, ct);
                if (approval is null || approval.Decision != OperatorDecision.Approved || string.IsNullOrWhiteSpace(approval.OperatorId) ||
                    approval.RequestId != approvalId || approval.Expires <= clock.GetUtcNow() ||
                    approval.RigId != context.RigId || approval.ChatId != context.ChatId || approval.Tool != name ||
                    Digest(EffectiveArguments(name, approval.ReviewedArguments)) != item.Digest ||
                    item.Preview.Expires <= clock.GetUtcNow())
                    throw new ArgumentException("Authenticated explicit approval for this exact operation is required.");
                Validate(name, arguments);
                ValidateContext(name, arguments, context);
                if (item.State != StateDigest(context)) throw new ArgumentException("State changed during approval verification.");
            }
            // Clone before awaiting so caller mutation cannot alter authorized parameters.
            return Normalize(await client.CallAsync(name, (JsonObject)arguments.DeepClone(), ct));
        }
        finally { calls.Release(); }
    }

    // Called directly by the authenticated host control, without inference.
    // It uses the same explicit, exact, single-use application approval.
    public Task<JsonObject> EmergencyStopAsync(McpClient client, RigContext context, string approvalId, CancellationToken ct = default) =>
        InvokeAsync(client, "stop_all", new(), context, approvalId, ct);

    public async Task<JsonObject> InvokeAndPollAsync(McpClient client, string name, JsonObject arguments, RigContext context,
        string? approvalId = null, CancellationToken ct = default) {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromSeconds(30));
        var result = await InvokeAsync(client, name, arguments, context, approvalId, deadline.Token);
        var jobId = result["job"]?["id"]?.GetValue<string>();
        for (var poll = 0; poll < 8 && result["requiresPolling"]?.GetValue<bool>() == true; poll++) {
            if (jobId is null) throw new AiUnavailableException("Running job omitted its identifier.");
            await Task.Delay(250, deadline.Token);
            result = await InvokeAsync(client, "get_job", new JsonObject { ["job_id"] = jobId, ["wait_seconds"] = 0 }, context, ct: deadline.Token);
            if (result["job"]?["id"]?.GetValue<string>() != jobId) throw new AiUnavailableException("MCP polling changed job identity.");
        }
        return result; // Still running is explicitly incomplete; never repeat the initial mutation.
    }

    private void ValidateContext(string name, JsonObject arguments, RigContext context) {
        if (string.IsNullOrWhiteSpace(context.RigId) || string.IsNullOrWhiteSpace(context.ChatId)) throw new ArgumentException("Rig/chat identity is required.");
        if (Reads.Contains(name)) return;
        var age = clock.GetUtcNow() - context.ObservedAt;
        if (age < TimeSpan.Zero || age > TimeSpan.FromSeconds(10)) throw new ArgumentException("Fresh verified rig state is required.");
        if (context.SequenceRunning && name is not ("stop_all" or "stop_sequence" or "cancel_job")) throw new ArgumentException("Sequence is running; AI mutation refused.");
        if (name == "set_switch") {
            var index = arguments["index"]!.GetValue<int>(); var value = arguments["value"]!.GetValue<double>();
            if (context.SwitchCapabilities is null || !context.SwitchCapabilities.TryGetValue(index, out var capability) ||
                !capability.Writable || string.IsNullOrWhiteSpace(capability.DeviceId) ||
                capability.Minimum > capability.Maximum || capability.Minimum != context.SwitchRanges.GetValueOrDefault(index).Minimum ||
                capability.Maximum != context.SwitchRanges.GetValueOrDefault(index).Maximum ||
                context.SwitchSteps?.GetValueOrDefault(index, double.NaN) != capability.Step)
                throw new ArgumentException("A verified writable channel and device identity are required.");
            if (!context.SwitchRanges.TryGetValue(index, out var range) || !double.IsFinite(range.Minimum) || !double.IsFinite(range.Maximum) || value < range.Minimum || value > range.Maximum)
                throw new ArgumentException("Switch range is unknown or value is outside live limits.");
            if (context.SwitchSteps is null || !context.SwitchSteps.TryGetValue(index, out var step) || !double.IsFinite(step) || step < 0)
                throw new ArgumentException("Verified writable-switch step size is required.");
            if (step > 0) {
                var steps = (value - range.Minimum) / step;
                if (!double.IsFinite(steps) || Math.Abs(steps - Math.Round(steps)) > 1e-6)
                    throw new ArgumentException("Switch value does not match its live step size.");
            }
        }
    }

    private static string StateDigest(RigContext context) {
        if (string.IsNullOrWhiteSpace(context.StateRevision) || context.DeviceIds is null || context.DeviceIds.Count == 0 ||
            context.DeviceIds.Any(d => string.IsNullOrWhiteSpace(d.Value)))
            throw new ArgumentException("Trusted state revision and device identities are required.");
        return Digest(new JsonObject {
            ["revision"] = context.StateRevision,
            ["rig"] = context.RigId,
            ["chat"] = context.ChatId,
            ["running"] = context.SequenceRunning,
            ["devices"] = System.Text.Json.JsonSerializer.SerializeToNode(context.DeviceIds),
            ["switches"] = System.Text.Json.JsonSerializer.SerializeToNode(context.SwitchCapabilities)
        });
    }

    private JsonObject EffectiveArguments(string name, JsonObject arguments) {
        var copy = JsonNode.Parse(arguments.ToJsonString())!.AsObject();
        // The server defaults many operations to a 30-second wait. Request a job
        // immediately instead, and bind this effective value to the operator preview.
        if (!copy.ContainsKey("wait_seconds") && catalog.TryGetValue(name, out var tool) &&
            tool["inputSchema"]?["properties"]?["wait_seconds"] is JsonObject)
            copy["wait_seconds"] = 0;
        return copy;
    }

    public static JsonObject Normalize(JsonObject envelope) {
        if (envelope["isError"]?.GetValue<bool>() == true) return new() { ["isError"] = true, ["message"] = "MCP reported an error; no success confirmed." };
        JsonObject? payload = envelope["structuredContent"] as JsonObject;
        if (payload is null && envelope["content"] is JsonArray content) {
            var text = content.FirstOrDefault(x => x?["type"]?.GetValue<string>() == "text")?["text"]?.GetValue<string>();
            if (text is not null) { try { payload = JsonNode.Parse(text) as JsonObject; } catch (System.Text.Json.JsonException) { } }
        }
        if (payload is null) return new() { ["isError"] = true, ["message"] = "No structured equipment result; completion unknown." };
        var result = (JsonObject)payload.DeepClone();
        result.Remove("next"); // MCP prose is data, never a continuation instruction.
        result["completionConfirmed"] = false;
        result["requiresPolling"] = false;
        result["resultTrust"] = "untrusted-equipment-data";
        if (result["success"] is JsonValue success && success.TryGetValue<bool>(out var ok) && !ok) result["isError"] = true;
        if (result["job"] is JsonObject job) {
            var state = job["state"]?.GetValue<string>();
            result["completionConfirmed"] = state == "succeeded" && job["outcome"]?["success"]?.GetValue<bool>() == true;
            result["requiresPolling"] = state is "running" or "queued";
            result["isError"] = state is "failed" or "cancelled" || state == "succeeded" && job["outcome"]?["success"]?.GetValue<bool>() != true;
        }
        if (result.ContainsKey("stopped") || result.ContainsKey("steps")) {
            result.Remove("stopped");
            result["stopRequested"] = true;
            result["requiresStatusVerification"] = true;
            result["isError"] = result["steps"] is not JsonArray steps || steps.Any(s => s?["success"]?.GetValue<bool>() != true);
        }
        // Never interpret nested text as further commands or approval.
        return result;
    }

    private static string Digest(JsonNode value) {
        JsonNode? Canonical(JsonNode? node) => node switch {
            JsonObject obj => new JsonObject(obj.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => KeyValuePair.Create(p.Key, Canonical(p.Value)))),
            JsonArray array => new JsonArray(array.Select(Canonical).ToArray()),
            _ => node?.DeepClone()
        };
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(Canonical(value)!.ToJsonString())));
    }

    private static void Schema(JsonNode? value, JsonObject schema) {
        if (schema["anyOf"] is JsonArray alternatives) {
            foreach (var alternative in alternatives) {
                try { Schema(value, alternative!.AsObject()); return; }
                catch (ArgumentException) { }
            }
            throw new ArgumentException("Tool argument matches no allowed schema variant.");
        }
        bool Matches(string type) => type switch {
            "object" => value is JsonObject,
            "array" => value is JsonArray,
            "string" => value is JsonValue v && v.TryGetValue<string>(out _),
            "boolean" => value is JsonValue v && v.TryGetValue<bool>(out _),
            "number" => value is JsonValue v && v.TryGetValue<double>(out var n) && double.IsFinite(n),
            "integer" => value is JsonValue v && v.TryGetValue<double>(out var n) && double.IsFinite(n) && Math.Truncate(n) == n,
            _ => false
        };
        var types = schema["type"] is JsonArray variants ? variants.Select(v => v!.GetValue<string>()).ToArray() : [schema["type"]!.GetValue<string>()];
        if (!types.Any(Matches)) throw new ArgumentException("Tool argument has wrong type.");
        if (value is JsonObject obj) {
            var properties = schema["properties"]!.AsObject();
            if (obj.Any(p => !properties.ContainsKey(p.Key))) throw new ArgumentException("Unexpected tool argument.");
            foreach (var required in schema["required"] as JsonArray ?? []) if (!obj.ContainsKey(required!.GetValue<string>())) throw new ArgumentException("Required tool argument missing.");
            foreach (var property in obj) Schema(property.Value, properties[property.Key]!.AsObject());
        }
        if (value is JsonArray array) {
            if (schema["minItems"] is { } min && array.Count < min.GetValue<int>() || schema["maxItems"] is { } max && array.Count > max.GetValue<int>()) throw new ArgumentException("Tool array size out of range.");
            foreach (var item in array) Schema(item, schema["items"]!.AsObject());
        }
        if (value is JsonValue scalar && scalar.TryGetValue<double>(out var number)) {
            if (schema["minimum"] is { } min && number < min.GetValue<double>() || schema["maximum"] is { } max && number > max.GetValue<double>() || schema["exclusiveMaximum"] is { } exclusive && number >= exclusive.GetValue<double>())
                throw new ArgumentException("Tool argument out of range.");
        }
        if (value is JsonValue text && text.TryGetValue<string>(out var str)) {
            if (schema["minLength"] is { } min && str.Length < min.GetValue<int>() || schema["maxLength"] is { } max && str.Length > max.GetValue<int>()) throw new ArgumentException("Tool string size out of range.");
            if (schema["pattern"] is { } pattern && !Regex.IsMatch(str, pattern.GetValue<string>(), RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(50))) throw new ArgumentException("Tool string pattern mismatch.");
            if (schema["enum"] is JsonArray choices && !choices.Any(c => c!.GetValue<string>() == str)) throw new ArgumentException("Invalid tool enum.");
        }
    }
}
