using System.Globalization;
using System.Text.Json.Nodes;

namespace Pins.AI.Runtime;

// Equipment limits come from structured MCP queries; profile/device/session
// identities come separately from the host, never from device names or prose.
public static class McpRigStateReader {
    public static async Task<RigContext> RefreshAsync(McpClient client, ToolGateway gateway,
        RigContext hostIdentity, TimeProvider? time = null, CancellationToken ct = default) {
        var status = await gateway.QueryStateAsync(client, "get_status", new(), ct);
        var sequence = await gateway.QueryStateAsync(client, "get_sequence", new() { ["max_depth"] = 1 }, ct);
        return Parse(status, sequence, hostIdentity, time ?? TimeProvider.System);
    }

    public static RigContext Parse(JsonObject status, JsonObject sequence, RigContext hostIdentity, TimeProvider time) {
        try {
            status = JsonNode.Parse(status.ToJsonString())!.AsObject();
            sequence = JsonNode.Parse(sequence.ToJsonString())!.AsObject();
            if (status["isError"]?.GetValue<bool>() == true || sequence["isError"]?.GetValue<bool>() == true ||
                hostIdentity.DeviceIds is null || string.IsNullOrWhiteSpace(hostIdentity.StateRevision))
                throw new ArgumentException("Trusted host identity and successful state queries are required.");
            var observed = DateTimeOffset.Parse(status["time"]!.GetValue<string>(), CultureInfo.InvariantCulture);
            var age = time.GetUtcNow() - observed;
            if (age < TimeSpan.FromSeconds(-1) || age > TimeSpan.FromSeconds(10))
                throw new ArgumentException("Equipment snapshot is stale or its clock is inconsistent.");
            var running = status["sequence"]!["running"]!.GetValue<bool>();
            if (sequence["initialized"]?.GetValue<bool>() != true || sequence["running"]?.GetValue<bool>() != running)
                throw new ArgumentException("Sequence state is uninitialized or inconsistent; mutation refused.");
            var ranges = new Dictionary<int, (double, double)>();
            var steps = new Dictionary<int, double>();
            var capabilities = new Dictionary<int, SwitchCapability>();
            var devices = status["devices"]!.AsArray();
            if (devices.Count == 0) throw new ArgumentException("Device status is unavailable.");
            foreach (var device in devices) {
                var properties = device!["properties"]!.AsObject();
                if (properties.ContainsKey("error")) throw new ArgumentException("Device status contains an error; mutation refused.");
                if (device["device"]!.GetValue<string>() != "switch" || !device["connected"]!.GetValue<bool>()) continue;
                if (!hostIdentity.DeviceIds.TryGetValue("switch", out var deviceId) || string.IsNullOrWhiteSpace(deviceId))
                    throw new ArgumentException("Verified switch hub identity is missing.");
                foreach (var channel in properties["switches"]!.AsArray()) {
                    if (channel!["writable"]?.GetValue<bool>() != true) continue;
                    var index = channel["index"]!.GetValue<int>();
                    var min = channel["min"]!.GetValue<double>();
                    var max = channel["max"]!.GetValue<double>();
                    var step = channel["step"]!.GetValue<double>();
                    if (index < 0 || !double.IsFinite(min) || !double.IsFinite(max) || !double.IsFinite(step) || min > max || step < 0)
                        throw new ArgumentException("Invalid live switch capabilities.");
                    ranges.Add(index, (min, max));
                    steps.Add(index, step);
                    capabilities.Add(index, new(deviceId + "/channel-" + index, true, min, max, step, channel["name"]?.GetValue<string>()));
                }
            }
            return hostIdentity with {
                SequenceRunning = running,
                ObservedAt = time.GetUtcNow(),
                SwitchRanges = ranges,
                SwitchSteps = steps,
                SwitchCapabilities = capabilities
            };
        }
        catch (Exception ex) when (ex is InvalidOperationException or NullReferenceException or FormatException) {
            throw new ArgumentException("Incomplete or invalid structured equipment state; mutation refused.", ex);
        }
    }
}
