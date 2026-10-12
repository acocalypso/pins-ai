using System.Text.Json;

namespace Pins.AI.Runtime;

public sealed record AiConfiguration {
    public bool Enabled { get; init; }
    public bool LazyLoad { get; init; } = true;
    public string Mode { get; init; } = "lite";
    public string ModelId { get; init; } = "qwen3-0.6b-q4";
    public string ModelPath { get; init; } = "";
    public string ModelSha256 { get; init; } = "";
    public string LoraPath { get; init; } = "";
    public string LoraSha256 { get; init; } = "";
    public string RuntimePath { get; init; } = "";
    public string LauncherPath { get; init; } = "";
    public int ContextTokens { get; init; } = 2048;
    public int Threads { get; init; } = 2;
    public int MaxOutputTokens { get; init; } = 128;
    public double Temperature { get; init; } = 0.2;
    public int InferenceTimeoutSeconds { get; init; } = 60;
    public int StartupTimeoutSeconds { get; init; } = 90;
    public int MinAvailableMemoryMiB { get; init; } = 768;
    public int MaxResidentMemoryMiB { get; init; } = 1200;
    public int MaxProcessSwapMiB { get; init; } = 64;
    public int MaxTemperatureC { get; init; } = 80;
    public int ApiPort { get; init; } = 4783;
    public bool McpChatEnabled { get; init; }
    public int McpChatPort { get; init; } = 4784;
    // Explicitly trusted browser origins for device-local testing; empty by default.
    public string[] McpChatTrustedOrigins { get; init; } = [];
    public string[] McpChatTrustedClients { get; init; } = [];
    public bool IsTrustedChatOrigin(string? origin) => origin is not null && McpChatTrustedOrigins.Contains(origin, StringComparer.Ordinal);
    public string LogLevel { get; init; } = "Warning";

    public static JsonSerializerOptions Json { get; } = new(JsonSerializerDefaults.Web) {
        WriteIndented = true,
        UnmappedMemberHandling = System.Text.Json.Serialization.JsonUnmappedMemberHandling.Disallow
    };

    public static string DefaultDataDirectory => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "pins-ai");

    public void Validate() {
        if (Mode is not ("lite" or "standard")) throw new ArgumentException("Mode must be lite or standard.");
        if (ContextTokens is < 512 or > 4096) throw new ArgumentException("Context must be 512–4096 tokens.");
        if (Threads is < 1 or > 3) throw new ArgumentException("Use 1–3 threads to leave capacity for PINS.");
        if (MaxOutputTokens is < 8 or > 256) throw new ArgumentException("Output limit must be 8–256 tokens.");
        if (!double.IsFinite(Temperature) || Temperature is < 0 or > 1) throw new ArgumentException("Invalid temperature.");
        if (InferenceTimeoutSeconds is < 5 or > 180 || StartupTimeoutSeconds is < 5 or > 180) throw new ArgumentException("Invalid timeout.");
        if (MinAvailableMemoryMiB is < 512 or > 2048 || MaxResidentMemoryMiB is < 256 or > 1536 || MaxProcessSwapMiB is < 0 or > 128)
            throw new ArgumentException("Invalid physical-memory or swap safeguards.");
        if (ApiPort is < 1024 or > 65535) throw new ArgumentException("Invalid API port.");
        if (McpChatPort is < 1024 or > 65535 || McpChatPort == ApiPort) throw new ArgumentException("Invalid MCP chat port.");
        foreach (var address in McpChatTrustedClients) {
            if (!System.Net.IPAddress.TryParse(address, out _)) throw new ArgumentException("Trusted chat clients must be exact IP addresses.");
        }
        foreach (var origin in McpChatTrustedOrigins) {
            if (!Uri.TryCreate(origin, UriKind.Absolute, out var uri) || uri.Scheme is not ("http" or "https") ||
                origin != uri.GetLeftPart(UriPartial.Authority) || !string.IsNullOrEmpty(uri.UserInfo))
                throw new ArgumentException("Trusted chat origins must be exact HTTP origins without credentials or paths.");
        }
        if (MaxTemperatureC is < 60 or > 85) throw new ArgumentException("Temperature cutoff must be 60–85 Celsius.");
        if (LogLevel is not ("Debug" or "Information" or "Warning" or "Error")) throw new ArgumentException("Invalid log level.");
        if (string.IsNullOrWhiteSpace(ModelId) || ModelId.Length > 128) throw new ArgumentException("Invalid model identifier.");
        if (Enabled) {
            foreach (var path in new[] { ModelPath, RuntimePath, LauncherPath }) {
                if (string.IsNullOrWhiteSpace(path) || !Path.IsPathFullyQualified(path)) throw new ArgumentException("Enabled configuration requires absolute model, runtime, launcher paths.");
            }
            if (ModelSha256.Length != 64 || !ModelSha256.All(Uri.IsHexDigit)) throw new ArgumentException("A verified model SHA256 is required.");
        }
        if (!string.IsNullOrEmpty(LoraPath) || !string.IsNullOrEmpty(LoraSha256)) {
            if (!Path.IsPathFullyQualified(LoraPath) || LoraSha256.Length != 64 || !LoraSha256.All(Uri.IsHexDigit))
                throw new ArgumentException("LoRA requires an absolute local path and verified SHA256.");
        }
    }

    public static async Task<AiConfiguration> ReadAsync(string path, CancellationToken ct = default) {
        var config = JsonSerializer.Deserialize<AiConfiguration>(await File.ReadAllTextAsync(path, ct), Json)
            ?? throw new ArgumentException("Empty configuration.");
        config.Validate();
        return config;
    }
}
