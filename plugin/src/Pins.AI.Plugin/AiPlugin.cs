using System.ComponentModel.Composition;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text;
using NINA.Profile.Interfaces;
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
    private McpChatApi? chatApi;
    private readonly IProfileService profileService;
    private readonly string instance = Guid.NewGuid().ToString("N");

    [ImportingConstructor]
    public AiPlugin(IProfileService profileService) { this.profileService = profileService; }

    private RigContext CaptureIdentity(string chatId) {
        var profile = profileService.ActiveProfile;
        var deviceIds = new Dictionary<string, string> {
            ["camera"] = profile.CameraSettings.Id ?? "No_Device",
            ["mount"] = profile.TelescopeSettings.Id ?? "No_Device",
            ["focuser"] = profile.FocuserSettings.Id ?? "No_Device",
            ["filter_wheel"] = profile.FilterWheelSettings.Id ?? "No_Device",
            ["guider"] = profile.GuiderSettings.GuiderName ?? "No_Device",
            ["rotator"] = profile.RotatorSettings.Id ?? "No_Device",
            ["flat_device"] = profile.FlatDeviceSettings.Id ?? "No_Device",
            ["dome"] = profile.DomeSettings.Id ?? "No_Device",
            ["switch"] = profile.SwitchSettings.Id ?? "No_Device",
            ["weather"] = profile.WeatherDataSettings.Id ?? "No_Device",
            ["safety_monitor"] = profile.SafetyMonitorSettings.Id ?? "No_Device"
        };
        // IDs come from the host profile, not MCP display names or request JSON.
        // Bind to all equipment settings, including static Alpaca addresses.
        var settings = new object[] { profile.CameraSettings, profile.TelescopeSettings, profile.FocuserSettings,
            profile.FilterWheelSettings, profile.GuiderSettings, profile.RotatorSettings, profile.FlatDeviceSettings,
            profile.DomeSettings, profile.SwitchSettings, profile.WeatherDataSettings, profile.SafetyMonitorSettings, profile.AlpacaSettings };
        var material = new StringBuilder(instance + profile.Id);
        foreach (var item in settings) {
            material.Append(item.GetType().FullName);
            foreach (var property in item.GetType().GetProperties().Where(p => p.CanRead && p.GetIndexParameters().Length == 0).OrderBy(p => p.Name)) {
                var value = property.GetValue(item);
                if (value is null || value is string || value.GetType().IsPrimitive || value is Enum || value is Guid)
                    material.Append(property.Name).Append('=').Append(Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture)).Append(';');
            }
        }
        var revision = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(material.ToString())));
        return new RigContext(profile.Id.ToString(), chatId, false, DateTimeOffset.UtcNow, new Dictionary<int, (double, double)>(),
            StateRevision: revision, DeviceIds: deviceIds);
    }

    public override async Task Initialize() {
        try {
            var data = AiConfiguration.DefaultDataDirectory;
            Directory.CreateDirectory(data);
            if (OperatingSystem.IsLinux()) File.SetUnixFileMode(data, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            var configPath = Path.Combine(data, "config.json");
            if (!File.Exists(configPath)) {
                var packagedDefault = "/usr/share/pins-ai/default-config.json";
                LlamaRuntime.WritePrivateFile(configPath, File.Exists(packagedDefault) ? await File.ReadAllTextAsync(packagedDefault) :
                    JsonSerializer.Serialize(new AiConfiguration(), AiConfiguration.Json));
            }
            var config = await AiConfiguration.ReadAsync(configPath);
            var keyPath = Path.Combine(data, "api.key");
            if (!File.Exists(keyPath)) LlamaRuntime.WritePrivateFile(keyPath, Convert.ToHexString(RandomNumberGenerator.GetBytes(32)));
            if (OperatingSystem.IsLinux()) File.SetUnixFileMode(keyPath, UnixFileMode.UserRead | UnixFileMode.UserWrite);
            runtime = new LlamaRuntime(config, data);
            api = new LocalApi(runtime, configPath, (await File.ReadAllTextAsync(keyPath)).Trim());
            api.Start();
            if (config.McpChatEnabled) {
                var mcpToken = Environment.GetEnvironmentVariable("PINS_MCP_TOKEN") ?? "";
                chatApi = new McpChatApi(runtime, mcpToken, CaptureIdentity);
                chatApi.Start();
            }
            Logger.Info($"PINS AI initialized: loopback port {config.ApiPort}, enabled={config.Enabled}; MCP chat enabled={config.McpChatEnabled}.");
            // An optional preload failure must not prevent the imaging host starting.
            try { await runtime.InitializeAsync(); } catch (Exception ex) { Logger.Warning($"PINS AI preload failed: {ex.Message}"); }
        }
        catch (Exception ex) {
            Logger.Error("PINS AI initialization failed; imaging remains independent.", ex);
            await Teardown();
        }
    }

    public override async Task Teardown() {
        if (chatApi is not null) { await chatApi.DisposeAsync(); chatApi = null; }
        if (api is not null) { await api.DisposeAsync(); api = null; }
        if (runtime is not null) { await runtime.DisposeAsync(); runtime = null; }
        Logger.Info("PINS AI stopped; owned inference process released.");
    }
}
