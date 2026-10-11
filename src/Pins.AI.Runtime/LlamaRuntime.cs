using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Pins.AI.Runtime;

public sealed class LlamaRuntime : IAsyncDisposable {
    private readonly SemaphoreSlim operation = new(1, 1);
    private readonly SemaphoreSlim management = new(1, 1);
    private readonly HttpClient http = new(new HttpClientHandler { UseProxy = false }) { Timeout = Timeout.InfiniteTimeSpan };
    private readonly string dataDirectory;
    private readonly ConcurrentQueue<string> logs = new();
    private CancellationTokenSource stop = new();
    private CancellationTokenSource? monitorStop;
    private Task? monitor;
    private Process? process;
    private Uri? endpoint;
    private string? runtimeKeyFile;
    private int state;
    private string? error;
    private bool disposed;
    public AiConfiguration Configuration { get; private set; }

    public LlamaRuntime(AiConfiguration configuration, string dataDirectory) {
        configuration.Validate();
        Configuration = configuration;
        this.dataDirectory = dataDirectory;
        Directory.CreateDirectory(dataDirectory);
        if (!OperatingSystem.IsLinux()) state = (int)ModelState.Unavailable;
    }

    private int? Pid {
        get { try { return process is { HasExited: false } p ? p.Id : null; } catch (InvalidOperationException) { return null; } }
    }

    public AiStatus Status => new(((ModelState)Volatile.Read(ref state)).ToString().ToLowerInvariant(),
        Configuration.Enabled, Configuration.ModelId, Pid, error, LinuxMemory.Read(Pid));
    public string[] Diagnostics => logs.ToArray();
    private void SetState(ModelState value) => Volatile.Write(ref state, (int)value);

    public async Task InitializeAsync(CancellationToken ct = default) {
        if (Configuration.Enabled && !Configuration.LazyLoad) await LoadAsync(ct);
    }

    public async Task LoadAsync(CancellationToken ct = default) {
        if (!await operation.WaitAsync(0, ct)) throw new AiBusyException();
        try {
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, stop.Token);
            await EnsureLoadedAsync(linked.Token);
        }
        finally { operation.Release(); }
    }

    private async Task EnsureLoadedAsync(CancellationToken ct) {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (!Configuration.Enabled) throw new AiUnavailableException("AI is disabled.");
        if (!OperatingSystem.IsLinux()) throw new AiUnavailableException("This runtime requires Linux.");
        if (Pid is not null && (ModelState)state == ModelState.Ready) return;
        await CleanupProcessAsync();
        error = null;
        SetState(ModelState.Loading);
        try {
            foreach (var path in new[] { Configuration.ModelPath, Configuration.RuntimePath, Configuration.LauncherPath })
                if (!File.Exists(path)) throw new FileNotFoundException("Configured model/runtime/launcher is missing.", path);
            if (!LinuxMemory.WithinBudget(Configuration, LinuxMemory.Read())) throw new AiUnavailableException("Insufficient available physical RAM.");
            if (LinuxMemory.TemperatureC >= Configuration.MaxTemperatureC) throw new AiUnavailableException("Pi is above the configured temperature cutoff; let it cool before inference.");
            await using (var model = File.OpenRead(Configuration.ModelPath)) {
                var hash = Convert.ToHexString(await SHA256.HashDataAsync(model, ct));
                if (!hash.Equals(Configuration.ModelSha256, StringComparison.OrdinalIgnoreCase)) throw new AiUnavailableException("Model SHA256 verification failed.");
            }
            // Ephemeral port, never attach to an unrelated process occupying it.
            var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            var port = ((IPEndPoint)listener.LocalEndpoint).Port;
            listener.Stop();
            endpoint = new Uri($"http://127.0.0.1:{port}/");
            var key = Convert.ToHexString(RandomNumberGenerator.GetBytes(32));
            runtimeKeyFile = Path.Combine(dataDirectory, $"runtime-{Guid.NewGuid():N}.key");
            WritePrivateFile(runtimeKeyFile, key + "\n");
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", key);
            var start = new ProcessStartInfo(Configuration.LauncherPath) {
                UseShellExecute = false,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
                WorkingDirectory = dataDirectory
            };
            string[] args = [Environment.ProcessId.ToString(System.Globalization.CultureInfo.InvariantCulture),
                Configuration.RuntimePath, "--model", Configuration.ModelPath,
                "--host", "127.0.0.1", "--port", port.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--api-key-file", runtimeKeyFile, "--ctx-size", Configuration.ContextTokens.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--threads", Configuration.Threads.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--threads-batch", Configuration.Threads.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--parallel", "1", "--n-gpu-layers", "0", "--batch-size", "128", "--ubatch-size", "64",
                "--jinja", "--chat-template-kwargs", "{\"enable_thinking\":false}", "--reasoning", "off", "--no-webui",
                "--log-verbosity", Configuration.LogLevel switch { "Debug" => "3", "Information" => "3", "Warning" => "2", _ => "1" }];
            foreach (var arg in args) start.ArgumentList.Add(arg);
            // Do not inherit llama environment overrides (especially remote tools).
            foreach (var name in start.Environment.Keys.Where(k => k.StartsWith("LLAMA_", StringComparison.Ordinal)).ToArray()) start.Environment.Remove(name);
            process = new Process { StartInfo = start };
            process.OutputDataReceived += CaptureLog;
            process.ErrorDataReceived += CaptureLog;
            if (!process.Start()) throw new AiUnavailableException("Could not start llama.cpp.");
            process.BeginOutputReadLine(); process.BeginErrorReadLine();
            monitorStop = new CancellationTokenSource();
            monitor = MonitorAsync(process, monitorStop.Token);
            using var startup = CancellationTokenSource.CreateLinkedTokenSource(ct, stop.Token);
            startup.CancelAfter(TimeSpan.FromSeconds(Configuration.StartupTimeoutSeconds));
            while (true) {
                startup.Token.ThrowIfCancellationRequested();
                if (process.HasExited) throw new AiUnavailableException("llama.cpp exited during startup; inspect bounded diagnostics.");
                try {
                    using var response = await http.GetAsync(new Uri(endpoint, "health"), startup.Token);
                    if (response.IsSuccessStatusCode) break;
                }
                catch (HttpRequestException) { }
                await Task.Delay(200, startup.Token);
            }
            SetState(ModelState.Ready);
        }
        catch (Exception ex) {
            error ??= ex is OperationCanceledException ? "Model load was cancelled or timed out." : ex.Message;
            await CleanupProcessAsync();
            SetState(ModelState.Failed);
            throw;
        }
    }

    private void CaptureLog(object sender, DataReceivedEventArgs args) {
        if (args.Data is not { } line) return;
        // Default llama logs only; redact credentials and retain no unbounded lines.
        if (line.Contains("api key", StringComparison.OrdinalIgnoreCase) || line.Contains("Authorization", StringComparison.OrdinalIgnoreCase)) return;
        logs.Enqueue(line.Length > 400 ? line[..400] : line);
        while (logs.Count > 40) logs.TryDequeue(out _);
    }

    private async Task MonitorAsync(Process owned, CancellationToken ct) {
        try {
            while (!ct.IsCancellationRequested && !owned.HasExited) {
                if (!LinuxMemory.WithinBudget(Configuration, LinuxMemory.Read(owned.Id))) {
                    error = "AI stopped: physical RAM, resident-memory, or process-swap safeguard exceeded.";
                    SetState(ModelState.Failed);
                    await stop.CancelAsync();
                    Kill(owned);
                    return;
                }
                if (LinuxMemory.TemperatureC >= Configuration.MaxTemperatureC) {
                    error = "AI stopped: configured temperature cutoff reached. Let the Pi cool before loading again.";
                    SetState(ModelState.Failed);
                    await stop.CancelAsync();
                    Kill(owned);
                    return;
                }
                await Task.Delay(500, ct);
            }
            if (!ct.IsCancellationRequested) {
                error = "llama.cpp exited unexpectedly. Explicit load or configuration reload is required.";
                SetState(ModelState.Failed);
                await stop.CancelAsync();
            }
        }
        catch (OperationCanceledException) { }
        catch (InvalidOperationException) { }
    }

    public async IAsyncEnumerable<TokenChunk> GenerateAsync(ChatRequest request,
        [EnumeratorCancellation] CancellationToken ct = default) {
        ChatPolicy.Validate(request);
        if (!await operation.WaitAsync(0, ct)) throw new AiBusyException();
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, stop.Token);
        // Includes lazy load, prefill and decode in the total turn deadline.
        linked.CancelAfter(TimeSpan.FromSeconds(Configuration.InferenceTimeoutSeconds));
        var success = false;
        try {
            await EnsureLoadedAsync(linked.Token);
            SetState(ModelState.Busy);
            // Qwen3.5's embedded template requires a single leading system message.
            var systemPrompt = ChatPolicy.SystemPrompt;
            if (!string.IsNullOrWhiteSpace(request.Locale)) systemPrompt += " Caller locale: " + request.Locale;
            var messages = new List<ChatMessage> { new("system", systemPrompt) };
            messages.AddRange(request.Messages);
            var payload = new {
                model = Configuration.ModelId,
                messages,
                stream = true,
                stream_options = new { include_usage = true },
                max_tokens = Configuration.MaxOutputTokens,
                temperature = Configuration.Temperature,
                seed = 42,
                cache_prompt = false,
                chat_template_kwargs = new { enable_thinking = false }
            };
            using var httpRequest = new HttpRequestMessage(HttpMethod.Post, new Uri(endpoint!, "v1/chat/completions")) {
                Content = new ByteArrayContent(JsonSerializer.SerializeToUtf8Bytes(payload, AiConfiguration.Json))
            };
            httpRequest.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var response = await http.SendAsync(httpRequest, HttpCompletionOption.ResponseHeadersRead, linked.Token);
            if (!response.IsSuccessStatusCode) throw new AiUnavailableException($"Inference rejected request (HTTP {(int)response.StatusCode}); shorten the history or inspect runtime diagnostics.");
            await using var stream = await response.Content.ReadAsStreamAsync(linked.Token);
            using var reader = new StreamReader(stream, Encoding.UTF8);
            var outputChars = 0;
            while (await ReadBoundedLineAsync(reader, 65536, linked.Token) is { } line) {
                if (!line.StartsWith("data: ", StringComparison.Ordinal)) continue;
                var data = line[6..];
                if (data == "[DONE]") { success = true; yield return new("", true); break; }
                using var json = JsonDocument.Parse(data);
                if (json.RootElement.TryGetProperty("error", out _)) throw new AiUnavailableException("Runtime reported an inference error.");
                if (json.RootElement.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0) {
                    var choice = choices[0];
                    if (choice.TryGetProperty("delta", out var delta) && delta.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String) {
                        var text = content.GetString()!;
                        outputChars += text.Length;
                        if (outputChars > 16384) throw new AiUnavailableException("Inference output exceeded the character budget.");
                        if (text.Length > 0) yield return new(text);
                    }
                }
                if (json.RootElement.TryGetProperty("timings", out var timings)) yield return new("", Timings: timings.Clone());
                else if (json.RootElement.TryGetProperty("usage", out var usage)) yield return new("", Timings: usage.Clone());
            }
            if (!success) throw new AiUnavailableException("Incomplete inference stream.");
        }
        finally {
            // A cancelled/disconnected stream kills the child, releasing work immediately.
            if (!success) await CleanupProcessAsync();
            if (error is not null) SetState(ModelState.Failed);
            else SetState(success && Pid is not null ? ModelState.Ready : ModelState.Unloaded);
            operation.Release();
        }
    }

    public static async Task<string?> ReadBoundedLineAsync(StreamReader reader, int limit, CancellationToken ct) {
        var result = new StringBuilder();
        var buffer = new char[1];
        while (await reader.ReadAsync(buffer.AsMemory(), ct) != 0) {
            if (buffer[0] == '\n') return result.ToString().TrimEnd('\r');
            if (result.Length >= limit) throw new AiUnavailableException("SSE line exceeded the response budget.");
            result.Append(buffer[0]);
        }
        return result.Length == 0 ? null : result.ToString();
    }

    public async Task CancelAsync() => await StopAsync();

    public async Task StopAsync() {
        await management.WaitAsync();
        try { await StopCoreAsync(); } finally { management.Release(); }
    }

    private async Task StopCoreAsync() {
        await stop.CancelAsync();
        await operation.WaitAsync();
        try {
            await CleanupProcessAsync();
            stop.Dispose(); stop = new CancellationTokenSource();
            error = null;
            SetState(OperatingSystem.IsLinux() ? ModelState.Unloaded : ModelState.Unavailable);
        }
        finally { operation.Release(); }
    }

    public async Task ReconfigureAsync(AiConfiguration configuration, CancellationToken ct = default) {
        configuration.Validate();
        await management.WaitAsync(ct);
        try {
            await StopCoreAsync();
            Configuration = configuration;
            await InitializeAsync(ct);
        }
        finally { management.Release(); }
    }

    private async Task CleanupProcessAsync() {
        if (monitorStop is not null) {
            await monitorStop.CancelAsync();
            if (monitor is not null) await monitor;
            monitorStop.Dispose(); monitorStop = null; monitor = null;
        }
        if (process is { } owned) {
            Kill(owned);
            using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            try { await owned.WaitForExitAsync(deadline.Token); } catch (OperationCanceledException) { }
            owned.Dispose(); process = null;
        }
        if (runtimeKeyFile is not null) { File.Delete(runtimeKeyFile); runtimeKeyFile = null; }
        endpoint = null;
    }

    private static void Kill(Process owned) {
        try { if (!owned.HasExited) owned.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
    }

    public static void WritePrivateFile(string path, string content) {
        // Create with restrictive mode before writing any secret bytes.
        var options = new FileStreamOptions { Mode = FileMode.CreateNew, Access = FileAccess.Write };
        if (OperatingSystem.IsLinux()) options.UnixCreateMode = UnixFileMode.UserRead | UnixFileMode.UserWrite;
        using var stream = new FileStream(path, options);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        writer.Write(content);
    }

    public async ValueTask DisposeAsync() {
        if (disposed) return;
        disposed = true;
        await StopAsync();
        http.Dispose();
    }
}
