using System.Text.Json.Nodes;

namespace Pins.AI.Runtime;

// Composition belongs to the plugin/host, never to model or HTTP request JSON.
// The host supplies authenticated session validation and fresh mediator snapshots.
public sealed class ApplicationOperatorAuthority(
    Func<string, CancellationToken, Task<string?>> authenticateOperator,
    Func<string, string, CancellationToken, Task<RigContext>> captureState,
    TimeProvider? time = null) : IOperatorAuthority {
    private readonly TimeProvider clock = time ?? TimeProvider.System;
    private readonly Dictionary<string, (OperatorApproval Approval, string Session)> decisions = [];

    public Task<RigContext> RefreshAsync(string rigId, string chatId, CancellationToken ct) => captureState(rigId, chatId, ct);

    public async Task<OperatorApproval?> GetApprovalAsync(string requestId, CancellationToken ct) {
        ct.ThrowIfCancellationRequested();
        (OperatorApproval Approval, string Session) item;
        lock (decisions) {
            if (!decisions.Remove(requestId, out item)) return null;
        }
        // Approval cannot outlive a revoked/expired authenticated app session.
        var currentOperator = await authenticateOperator(item.Session, ct);
        return currentOperator == item.Approval.OperatorId && item.Approval.Expires > clock.GetUtcNow() ? item.Approval : null;
    }

    // Invoke from an explicit application approval event. Bearer chat access,
    // user prose and a tool result must never be accepted by authenticateOperator.
    public async Task DecideAsync(string authenticatedSession, ToolPreview preview, OperatorDecision decision,
        string requestedTool, JsonObject requestedArguments, bool draftOnly = false, CancellationToken ct = default) {
        var operatorId = await authenticateOperator(authenticatedSession, ct);
        if (string.IsNullOrWhiteSpace(operatorId)) throw new ArgumentException("An authenticated operator session is required.");
        if (preview.Expires <= clock.GetUtcNow()) throw new ArgumentException("Preview expired.");
        if (decision == OperatorDecision.Approved)
            OperatorArguments.MatchIntent(preview.Tool, preview.Arguments, requestedTool, requestedArguments, draftOnly);
        var approval = new OperatorApproval(decision, preview.RequestId, operatorId, preview.RigId, preview.ChatId,
            preview.Tool, (JsonObject)preview.Arguments.DeepClone(),
            preview.Expires < clock.GetUtcNow().AddSeconds(30) ? preview.Expires : clock.GetUtcNow().AddSeconds(30));
        lock (decisions) {
            foreach (var expired in decisions.Where(d => d.Value.Approval.Expires <= clock.GetUtcNow()).Select(d => d.Key).ToArray()) decisions.Remove(expired);
            if (decisions.ContainsKey(preview.RequestId)) throw new ArgumentException("Decision already recorded; create a new preview.");
            if (decisions.Count >= 32) throw new ArgumentException("Too many pending operator decisions.");
            decisions.Add(preview.RequestId, (approval, authenticatedSession));
        }
    }
}
