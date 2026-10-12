using System.Collections.Concurrent;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Pins.AI.Runtime;

// Owned and disposed by the PINS plugin, never an independent service/process.
public sealed class McpChatApi : IAsyncDisposable {
    private readonly HttpListener listener = new();
    private readonly CancellationTokenSource stop = new();
    private readonly ConcurrentDictionary<long, Task> requests = new();
    private readonly SemaphoreSlim slots = new(8, 8);
    private readonly McpChatService service;
    private readonly LlamaRuntime runtime;
    private readonly byte[] credential;
    private Task? loop;
    private long sequence;
    public McpChatApi(LlamaRuntime runtime, string token, Func<string, RigContext> identity) {
        if (token.Length < 32) throw new ArgumentException("MCP chat requires a private MCP bearer token of at least 32 characters.");
        this.runtime = runtime;
        credential = Encoding.UTF8.GetBytes("Bearer " + token);
        service = new McpChatService(runtime, token, identity);
        listener.Prefixes.Add($"http://+:{runtime.Configuration.McpChatPort}/");
    }
    public void Start() { listener.Start(); loop = AcceptAsync(); }
    private async Task AcceptAsync() {
        try {
            while (!stop.IsCancellationRequested) {
                var context = await listener.GetContextAsync().WaitAsync(stop.Token);
                if (!await slots.WaitAsync(0, stop.Token)) { context.Response.StatusCode = 429; context.Response.Close(); continue; }
                var id = Interlocked.Increment(ref sequence);
                var task = HandleAsync(context); requests[id] = task;
                _ = task.ContinueWith(_ => { requests.TryRemove(id, out var ignored); slots.Release(); },
                    CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
            }
        }
        catch (OperationCanceledException) { }
        catch (HttpListenerException) when (stop.IsCancellationRequested) { }
        catch (ObjectDisposedException) when (stop.IsCancellationRequested) { }
    }
    private async Task HandleAsync(HttpListenerContext context) {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(stop.Token);
        deadline.CancelAfter(TimeSpan.FromSeconds(Math.Min(180, runtime.Configuration.StartupTimeoutSeconds + runtime.Configuration.InferenceTimeoutSeconds + 30)));
        var response = context.Response;
        try {
            response.Headers["Cache-Control"] = "no-store";
            response.Headers["X-Content-Type-Options"] = "nosniff";
            // Explicit trusted frontend origins use the device-owned MCP token.
            // Sessions and separate mutation approvals still apply.
            var origin = context.Request.Headers["Origin"];
            var address = context.Request.RemoteEndPoint?.Address;
            var clientAddress = address?.IsIPv4MappedToIPv6 == true ? address.MapToIPv4().ToString() : address?.ToString();
            var trustedOrigin = runtime.Configuration.IsTrustedChatOrigin(origin) &&
                runtime.Configuration.McpChatTrustedClients.Contains(clientAddress, StringComparer.Ordinal);
            response.Headers["Access-Control-Allow-Origin"] = trustedOrigin ? origin! : "*";
            response.Headers["Vary"] = "Origin";
            response.Headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-PINS-Chat-Session";
            response.Headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS";
            if (context.Request.HttpMethod == "OPTIONS") { response.StatusCode = 204; return; }
            var supplied = Encoding.UTF8.GetBytes(context.Request.Headers["Authorization"] ?? "");
            if (!trustedOrigin && (supplied.Length != credential.Length || !CryptographicOperations.FixedTimeEquals(supplied, credential))) {
                await JsonAsync(context, 401, new { error = "unauthorized" }); return;
            }
            var method = context.Request.HttpMethod;
            var path = context.Request.Url!.AbsolutePath;
            if (method == "GET" && path == "/status") {
                var status = await service.DiagnosticsAsync(deadline.Token);
                status["mcpChatAvailable"] = true;
                status["port"] = runtime.Configuration.McpChatPort;
                await JsonAsync(context, 200, status); return;
            }
            if (method == "GET" && path == "/diagnostics") {
                var diagnostics = await service.DiagnosticsAsync(deadline.Token);
                diagnostics["lines"] = JsonSerializer.SerializeToNode(runtime.Diagnostics, AiConfiguration.Json);
                diagnostics["gateway"] = "Exact schema validation; separate expiring operator approval; no mutation retries.";
                await JsonAsync(context, 200, diagnostics); return;
            }
            if (method == "POST" && path == "/session") {
                var body = await BodyAsync(context.Request, deadline.Token);
                if (body.Count != 0) throw new ArgumentException("Session request must be empty.");
                await JsonAsync(context, 200, await service.OpenAsync(deadline.Token)); return;
            }
            var session = context.Request.Headers["X-PINS-Chat-Session"] ?? "";
            if (method == "DELETE" && path == "/session") {
                await service.CloseAsync(session); await JsonAsync(context, 200, new { disconnected = true }); return;
            }
            if (method == "POST" && path is "/chat" or "/tool" or "/approve" or "/reject" or "/clear") {
                var body = await BodyAsync(context.Request, deadline.Token);
                var result = await service.HandleAsync(session, path[1..], body, deadline.Token);
                await JsonAsync(context, 200, result); return;
            }
            if (method == "POST" && path == "/cancel") {
                var body = await BodyAsync(context.Request, deadline.Token);
                if (body.Count != 0) throw new ArgumentException("Cancel request must be empty.");
                await service.CancelAsync(session); await JsonAsync(context, 200, new { inferenceCancelled = true, hardwareStopped = false }); return;
            }
            await JsonAsync(context, 404, new { error = "unknown_endpoint" });
        }
        catch (Exception ex) when (ex is ArgumentException or JsonException or InvalidOperationException or AiBusyException or AiUnavailableException or OperationCanceledException or HttpRequestException or IOException or HttpListenerException) {
            var status = ex is AiBusyException ? 409 : ex is ArgumentException or JsonException or InvalidOperationException ? 400 : ex is OperationCanceledException ? 408 : 503;
            try { await JsonAsync(context, status, new { error = ex is OperationCanceledException ? "Operation cancelled or timed out; inspect job status before retrying." : ex.Message }); }
            catch (Exception error) when (error is IOException or HttpListenerException or ObjectDisposedException) { }
        }
        finally { response.Close(); }
    }
    private static async Task<JsonObject> BodyAsync(HttpListenerRequest request, CancellationToken ct) {
        if (request.ContentType?.StartsWith("application/json", StringComparison.OrdinalIgnoreCase) != true || request.ContentLength64 > 32768)
            throw new ArgumentException("Supply bounded application/json.");
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct); deadline.CancelAfter(TimeSpan.FromSeconds(10));
        using var memory = new MemoryStream(); var buffer = new byte[4096]; int count;
        while ((count = await request.InputStream.ReadAsync(buffer, deadline.Token)) != 0) {
            if (memory.Length + count > 32768) throw new ArgumentException("Request exceeds 32 KiB.");
            memory.Write(buffer, 0, count);
        }
        return JsonNode.Parse(memory.ToArray(), documentOptions: new() { MaxDepth = 24 })?.AsObject() ?? throw new ArgumentException("Expected JSON object.");
    }
    private static async Task JsonAsync(HttpListenerContext context, int status, object body) {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(body, AiConfiguration.Json);
        context.Response.StatusCode = status; context.Response.ContentType = "application/json; charset=utf-8";
        context.Response.ContentLength64 = bytes.Length;
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await context.Response.OutputStream.WriteAsync(bytes, deadline.Token);
    }
    public async ValueTask DisposeAsync() {
        await stop.CancelAsync(); listener.Close();
        if (loop is not null) await loop;
        await runtime.CancelAsync();
        await Task.WhenAll(requests.Values);
        await service.DisposeAsync(); stop.Dispose();
    }
}
