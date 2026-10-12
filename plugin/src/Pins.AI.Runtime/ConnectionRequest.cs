using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Pins.AI.Runtime;

// Exact connection instructions become proposals only. Unknown/qualified text
// is left to inference; never extract an action from a question or a negation.
public static class ConnectionRequest {
    private static readonly Dictionary<string, string> Devices = new(StringComparer.OrdinalIgnoreCase) {
        ["camera"] = "camera",
        ["kamera"] = "camera",
        ["mount"] = "mount",
        ["montierung"] = "mount",
        ["focuser"] = "focuser",
        ["fokussierer"] = "focuser",
        ["filter wheel"] = "filter_wheel",
        ["filterrad"] = "filter_wheel",
        ["guider"] = "guider",
        ["rotator"] = "rotator",
        ["dome"] = "dome",
        ["kuppel"] = "dome"
    };
    public static JsonObject[]? Parse(string message) {
        var match = Regex.Match(message.Trim(), @"\A(connect|disconnect|verbinde|verbinden|trenne|trennen)\s+(.+?)[.!]?\z",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100));
        if (!match.Success) return null;
        var names = Regex.Split(match.Groups[2].Value, @"\s*(?:,|\band\b|\bund\b)\s*",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100));
        if (names.Length is < 1 or > 4) return null;
        var devices = new List<string>();
        foreach (var name in names) {
            var bare = Regex.Replace(name.Trim(), @"\A(?:(?:the|die|den|das|selected)\s+)+", "",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100));
            if (!Devices.TryGetValue(bare, out var device) || devices.Contains(device)) return null;
            devices.Add(device);
        }
        var action = match.Groups[1].Value.ToLowerInvariant() is "disconnect" or "trenne" or "trennen" ? "disconnect_device" : "connect_device";
        return devices.Select(device => new JsonObject {
            ["name"] = action,
            ["arguments"] = new JsonObject { ["device"] = device, ["wait_seconds"] = 0 }
        }).ToArray();
    }
}
