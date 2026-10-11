using System.Globalization;
using System.Text.Json.Nodes;

namespace Pins.AI.Runtime;

public sealed record SwitchCapability(string DeviceId, bool Writable, double Minimum, double Maximum, double Step, string? Name = null);
public enum OperatorDecision { Pending, Denied, Approved }
public sealed record OperatorApproval(OperatorDecision Decision, string RequestId, string OperatorId,
    string RigId, string ChatId, string Tool, JsonObject ReviewedArguments, DateTimeOffset Expires);

// Implement inside the authenticated PINS application. Never deserialize this
// service or its decisions from chat/MCP/model JSON. It must verify the operator
// session and compare the original structured operator intent with the preview.
public interface IOperatorAuthority {
    Task<RigContext> RefreshAsync(string rigId, string chatId, CancellationToken ct);
    Task<OperatorApproval?> GetApprovalAsync(string requestId, CancellationToken ct);
}

public static class OperatorArguments {
    public static int NamedSwitch(string name, IReadOnlyDictionary<int, SwitchCapability> channels) {
        var matches = channels.Where(c => c.Value.Name == name).ToArray();
        if (string.IsNullOrWhiteSpace(name) || matches.Length != 1 || !matches[0].Value.Writable)
            throw new ArgumentException("Switch name is unknown, ambiguous or read-only; specify a verified writable channel.");
        return matches[0].Key;
    }

    // Only for explicit numeric fields in a trusted structured input form.
    // Thousands separators, sexagesimal coordinates and inferred values are refused.
    public static double Decimal(string text) {
        if (!System.Text.RegularExpressions.Regex.IsMatch(text, @"^[+-]?\d+(?:[.,]\d+)?$",
            System.Text.RegularExpressions.RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(50)) ||
            !double.TryParse(text.Replace(',', '.'), NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint,
                CultureInfo.InvariantCulture, out var value) || !double.IsFinite(value))
            throw new ArgumentException("An explicit decimal value is required; clarify ambiguous units or separators.");
        return value;
    }

    public static void MatchIntent(string proposedTool, JsonObject proposed, string requestedTool, JsonObject requested, bool draftOnly = false) {
        requested = (JsonObject)requested.DeepClone();
        // The gateway adds an immediate job return as a transport bound, not
        // an equipment argument. Permit only this exact zero-wait default.
        if (!requested.ContainsKey("wait_seconds") && proposed["wait_seconds"]?.ToJsonString() == "0") requested["wait_seconds"] = 0;
        if (draftOnly && proposedTool == "start_sequence" || proposedTool != requestedTool ||
            !JsonNode.DeepEquals(proposed, requested))
            throw new ArgumentException("Proposal differs from the operator's exact structured intent; clarify instead of substituting.");
    }
}
