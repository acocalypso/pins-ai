using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json.Nodes;

namespace Pins.AI.Runtime;

// Client for the team's JSON-response Streamable HTTP endpoint; never equipment APIs.
public sealed class McpClient : IAsyncDisposable {
    private readonly HttpClient http;
    private readonly Uri endpoint;
    private readonly SemaphoreSlim serial = new(1, 1);
    private string? session;
    private string protocol = "2025-11-25";
    private long sequence;
    public McpClient(Uri endpoint, string token, HttpMessageHandler? transport = null) {
        if (endpoint.Scheme != "http" || endpoint.Host != "127.0.0.1" || endpoint.UserInfo.Length != 0 || endpoint.Fragment.Length != 0)
            throw new ArgumentException("MCP must use explicit loopback HTTP.");
        if (token.Length < 32) throw new ArgumentException("Configure a private MCP bearer token.");
        this.endpoint = endpoint;
        http = new(transport ?? new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false }) { Timeout = TimeSpan.FromSeconds(15) };
        http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        http.DefaultRequestHeaders.Accept.ParseAdd("application/json");
        http.DefaultRequestHeaders.Accept.ParseAdd("text/event-stream");
    }

    public async Task<JsonObject> DiscoverAsync(CancellationToken ct = default) {
        await serial.WaitAsync(ct);
        try {
            if (session is null) {
                var result = await SendAsync("initialize", new JsonObject {
                    ["protocolVersion"] = protocol,
                    ["capabilities"] = new JsonObject(),
                    ["clientInfo"] = new JsonObject { ["name"] = "pins-ai", ["version"] = "0.2.0" }
                }, ct);
                var negotiated = result["protocolVersion"]?.GetValue<string>();
                if (negotiated is not ("2025-11-25" or "2025-06-18" or "2025-03-26") || session is null)
                    throw new AiUnavailableException("MCP initialization did not establish a supported session.");
                protocol = negotiated;
                await SendAsync("notifications/initialized", null, ct, notification: true);
            }
            return await SendAsync("tools/list", null, ct);
        }
        finally { serial.Release(); }
    }

    // Internal: every production tools/call must enter through ToolGateway.
    internal async Task<JsonObject> CallAsync(string name, JsonObject arguments, CancellationToken ct) {
        await serial.WaitAsync(ct);
        try {
            if (session is null) throw new AiUnavailableException("Discover MCP tools before calling them.");
            return await SendAsync("tools/call", new JsonObject { ["name"] = name, ["arguments"] = arguments.DeepClone() }, ct);
        }
        finally { serial.Release(); }
    }

    private async Task<JsonObject> SendAsync(string method, JsonObject? parameters, CancellationToken ct, bool notification = false) {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromSeconds(15));
        ct = deadline.Token; // ResponseHeadersRead does not bound the response body by itself.
        var id = Interlocked.Increment(ref sequence);
        var body = new JsonObject { ["jsonrpc"] = "2.0", ["method"] = method };
        if (!notification) body["id"] = id;
        if (parameters is not null) body["params"] = parameters;
        using var request = new HttpRequestMessage(HttpMethod.Post, endpoint) { Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json") };
        request.Headers.Add("MCP-Protocol-Version", protocol);
        if (session is not null) request.Headers.Add("Mcp-Session-Id", session);
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        if (response.StatusCode == HttpStatusCode.NotFound) session = null;
        if (!response.IsSuccessStatusCode) throw new AiUnavailableException($"MCP transport returned HTTP {(int)response.StatusCode}; no automatic tool retries.");
        if (method == "initialize" && response.Headers.TryGetValues("Mcp-Session-Id", out var sessions)) {
            var supplied = sessions.Single();
            if (supplied.Length is < 1 or > 256) throw new AiUnavailableException("Invalid MCP session identifier.");
            session = supplied;
        }
        if (notification && response.StatusCode == HttpStatusCode.Accepted) return new();
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var memory = new MemoryStream();
        var buffer = new byte[4096];
        int count;
        while ((count = await stream.ReadAsync(buffer, ct)) != 0) {
            if (memory.Length + count > 65536) throw new AiUnavailableException("MCP result exceeds 64 KiB.");
            memory.Write(buffer, 0, count);
        }
        var json = JsonNode.Parse(memory.ToArray(), documentOptions: new() { MaxDepth = 32 }) as JsonObject;
        if (json is null || json["jsonrpc"]?.GetValue<string>() != "2.0" || json["id"]?.GetValue<long>() != id || json.ContainsKey("error"))
            throw new AiUnavailableException("Invalid or unsuccessful MCP response.");
        return json["result"] as JsonObject ?? throw new AiUnavailableException("MCP result must be an object.");
    }

    public async ValueTask DisposeAsync() {
        if (session is not null) {
            try {
                using var request = new HttpRequestMessage(HttpMethod.Delete, endpoint);
                request.Headers.Add("Mcp-Session-Id", session);
                request.Headers.Add("MCP-Protocol-Version", protocol);
                using var response = await http.SendAsync(request);
            }
            catch (HttpRequestException) { }
            catch (OperationCanceledException) { }
        }
        http.Dispose(); serial.Dispose();
    }
}
