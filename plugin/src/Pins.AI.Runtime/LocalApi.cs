using System.Collections.Concurrent;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Pins.AI.Runtime;

public sealed class LocalApi : IAsyncDisposable {
    private readonly LlamaRuntime runtime;
    private readonly string configPath;
    private readonly byte[] token;
    private readonly HttpListener listener = new();
    private readonly CancellationTokenSource shutdown = new();
    private readonly ConcurrentDictionary<long, Task> requests = new();
    private readonly SemaphoreSlim slots = new(8, 8);
    private Task? acceptLoop;
    private long sequence;
    public LocalApi(LlamaRuntime runtime, string configPath, string apiToken) {
        this.runtime = runtime; this.configPath = configPath;
        if (apiToken.Length < 32) throw new ArgumentException("API token is too short.");
        token = Encoding.UTF8.GetBytes("Bearer " + apiToken);
        listener.Prefixes.Add($"http://127.0.0.1:{runtime.Configuration.ApiPort}/");
        listener.IgnoreWriteExceptions = false;
    }

    public void Start() { listener.Start(); acceptLoop = AcceptAsync(); }

    private async Task AcceptAsync() {
        try {
            while (!shutdown.IsCancellationRequested) {
                var context = await listener.GetContextAsync().WaitAsync(shutdown.Token);
                if (!await slots.WaitAsync(0, shutdown.Token)) {
                    context.Response.StatusCode = 429; context.Response.Close(); continue;
                }
                var id = Interlocked.Increment(ref sequence);
                var task = HandleAsync(context);
                requests[id] = task;
                _ = task.ContinueWith(completed => { requests.TryRemove(id, out _); slots.Release(); },
                    CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
            }
        }
        catch (OperationCanceledException) { }
        catch (HttpListenerException) when (shutdown.IsCancellationRequested) { }
        catch (ObjectDisposedException) when (shutdown.IsCancellationRequested) { }
    }

    private bool Authorized(HttpListenerRequest request) {
        if (request.RemoteEndPoint is not { } peer || !IPAddress.IsLoopback(peer.Address)) return false;
        var supplied = Encoding.UTF8.GetBytes(request.Headers["Authorization"] ?? "");
        return supplied.Length == token.Length && CryptographicOperations.FixedTimeEquals(supplied, token);
    }

    private async Task HandleAsync(HttpListenerContext context) {
        var streaming = false;
        using var requestDeadline = CancellationTokenSource.CreateLinkedTokenSource(shutdown.Token);
        requestDeadline.CancelAfter(TimeSpan.FromSeconds(runtime.Configuration.StartupTimeoutSeconds + 10));
        try {
            context.Response.Headers["Cache-Control"] = "no-store";
            context.Response.Headers["X-Content-Type-Options"] = "nosniff";
            if (!Authorized(context.Request)) {
                await JsonAsync(context, 401, new { error = "unauthorized" }); return;
            }
            var method = context.Request.HttpMethod;
            var path = context.Request.Url!.AbsolutePath;
            if (method == "GET" && path == "/status") {
                await JsonAsync(context, 200, runtime.Status);
            }
            else if (method == "GET" && path == "/diagnostics") {
                await JsonAsync(context, 200, new { lines = runtime.Diagnostics });
            }
            else if (method == "POST" && path == "/load") {
                await runtime.LoadAsync(shutdown.Token);
                await JsonAsync(context, 200, runtime.Status);
            }
            else if (method == "POST" && path is "/unload" or "/cancel") {
                await runtime.StopAsync();
                await JsonAsync(context, 200, runtime.Status);
            }
            else if (method == "POST" && path == "/reload") {
                var config = await AiConfiguration.ReadAsync(configPath, shutdown.Token);
                if (config.ApiPort != runtime.Configuration.ApiPort) throw new ArgumentException("Changing the API port requires plugin restart.");
                await runtime.ReconfigureAsync(config, shutdown.Token);
                await JsonAsync(context, 200, runtime.Status);
            }
            else if (method == "POST" && path == "/chat") {
                var request = await ReadChatAsync(context.Request, shutdown.Token);
                ChatPolicy.Validate(request);
                if (!runtime.Configuration.Enabled) throw new AiUnavailableException("AI is disabled.");
                streaming = true;
                requestDeadline.CancelAfter(TimeSpan.FromSeconds(runtime.Configuration.InferenceTimeoutSeconds));
                context.Response.StatusCode = 200;
                context.Response.ContentType = "text/event-stream; charset=utf-8";
                context.Response.SendChunked = true;
                await EventAsync(context, "capabilities", new { mcpAvailable = false, toolExecutionEnabled = false, informationalOnly = true }, requestDeadline.Token);
                await foreach (var chunk in runtime.GenerateAsync(request, requestDeadline.Token)) {
                    await EventAsync(context, chunk.Done ? "done" : chunk.Timings is not null ? "metrics" : "token", chunk, requestDeadline.Token);
                }
            }
            else await JsonAsync(context, 404, new { error = "unknown_endpoint" });
        }
        catch (Exception ex) when (ex is ArgumentException or JsonException or AiBusyException or AiUnavailableException or OperationCanceledException or HttpRequestException or IOException or HttpListenerException) {
            var status = ex is AiBusyException ? 409 : ex is ArgumentException or JsonException ? 400 : ex is OperationCanceledException ? 408 : 503;
            var message = ex is OperationCanceledException ? runtime.Status.Error ?? "AI operation cancelled or timed out." : ex.Message;
            try {
                if (streaming) await EventAsync(context, "error", new { status, error = message });
                else await JsonAsync(context, status, new { error = message });
            }
            catch (Exception writeError) when (writeError is IOException or HttpListenerException or OperationCanceledException or ObjectDisposedException) { }
        }
        finally { context.Response.Close(); }
    }

    private static async Task<ChatRequest> ReadChatAsync(HttpListenerRequest request, CancellationToken ct) {
        if (request.ContentLength64 > 16384) throw new ArgumentException("Request is too large.");
        if (request.ContentType?.StartsWith("application/json", StringComparison.OrdinalIgnoreCase) != true)
            throw new ArgumentException("Content-Type must be application/json.");
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromSeconds(10));
        using var content = new MemoryStream();
        var buffer = new byte[4096];
        int length;
        while ((length = await request.InputStream.ReadAsync(buffer, deadline.Token)) != 0) {
            if (content.Length + length > 16384) throw new ArgumentException("Request is too large.");
            await content.WriteAsync(buffer.AsMemory(0, length), deadline.Token);
        }
        return JsonSerializer.Deserialize<ChatRequest>(content.ToArray(), AiConfiguration.Json) ?? throw new ArgumentException("Empty chat request.");
    }

    private async Task JsonAsync(HttpListenerContext context, int status, object value) {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(value, AiConfiguration.Json);
        context.Response.StatusCode = status;
        context.Response.ContentType = "application/json; charset=utf-8";
        context.Response.ContentLength64 = bytes.Length;
        await context.Response.OutputStream.WriteAsync(bytes, shutdown.Token);
    }

    private async Task EventAsync(HttpListenerContext context, string name, object value, CancellationToken ct = default) {
        var json = JsonSerializer.Serialize(value, new JsonSerializerOptions(JsonSerializerDefaults.Web));
        var bytes = Encoding.UTF8.GetBytes($"event: {name}\ndata: {json}\n\n");
        // Await socket writes to preserve ordering and apply backpressure.
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct, shutdown.Token);
        timeout.CancelAfter(TimeSpan.FromSeconds(3));
        await context.Response.OutputStream.WriteAsync(bytes, timeout.Token);
        await context.Response.OutputStream.FlushAsync(timeout.Token);
    }

    public async ValueTask DisposeAsync() {
        await shutdown.CancelAsync();
        listener.Close();
        if (acceptLoop is not null) await acceptLoop;
        await runtime.StopAsync();
        await Task.WhenAll(requests.Values);
        shutdown.Dispose();
    }
}
