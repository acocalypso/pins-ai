using System.ComponentModel.Composition;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using NINA.Core.Utility;
using NINA.Plugin;
using NINA.Plugin.Interfaces;
using Pins.AI.Runtime;

[assembly: Guid("2bc537d9-2a43-4acd-b19c-d976f21ae7ea")]

namespace Pins.AI.Plugin;

[Export(typeof(IPluginManifest))]
public sealed class AiPlugin : PluginBase {
    private LlamaRuntime? runtime;
    private LocalApi? api;

    public override async Task Initialize() {
        try {
            var data = AiConfiguration.DefaultDataDirectory;
            Directory.CreateDirectory(data);
            if (OperatingSystem.IsLinux()) File.SetUnixFileMode(data, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            var configPath = Path.Combine(data, "config.json");
            if (!File.Exists(configPath)) LlamaRuntime.WritePrivateFile(configPath, JsonSerializer.Serialize(new AiConfiguration(), AiConfiguration.Json));
            var config = await AiConfiguration.ReadAsync(configPath);
            var keyPath = Path.Combine(data, "api.key");
            if (!File.Exists(keyPath)) LlamaRuntime.WritePrivateFile(keyPath, Convert.ToHexString(RandomNumberGenerator.GetBytes(32)));
            if (OperatingSystem.IsLinux()) File.SetUnixFileMode(keyPath, UnixFileMode.UserRead | UnixFileMode.UserWrite);
            runtime = new LlamaRuntime(config, data);
            api = new LocalApi(runtime, configPath, (await File.ReadAllTextAsync(keyPath)).Trim());
            api.Start();
            Logger.Info($"PINS AI initialized: loopback port {config.ApiPort}, enabled={config.Enabled}; MCP/tools unavailable.");
            // An optional preload failure must not prevent the imaging host starting.
            try { await runtime.InitializeAsync(); } catch (Exception ex) { Logger.Warning($"PINS AI preload failed: {ex.Message}"); }
        }
        catch (Exception ex) {
            Logger.Error("PINS AI initialization failed; imaging remains independent.", ex);
            await Teardown();
        }
    }

    public override async Task Teardown() {
        if (api is not null) { await api.DisposeAsync(); api = null; }
        if (runtime is not null) { await runtime.DisposeAsync(); runtime = null; }
        Logger.Info("PINS AI stopped; owned inference process released.");
    }
}
