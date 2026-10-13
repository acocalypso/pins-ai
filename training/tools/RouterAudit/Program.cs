using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Pins.AI.Runtime;

if (args.Length != 3) {
    Console.Error.WriteLine("Usage: dotnet run --project training/tools/RouterAudit -- <train.jsonl> <verified-tools-list.json> <output.json>");
    return 2;
}

try {
    var dataPath = Path.GetFullPath(args[0]);
    var catalogPath = Path.GetFullPath(args[1]);
    var outputPath = Path.GetFullPath(args[2]);
    if (outputPath == dataPath || outputPath == catalogPath) throw new ArgumentException("Output must not overwrite an input.");
    var catalogDocument = JsonNode.Parse(File.ReadAllText(catalogPath, Encoding.UTF8))?.AsObject()
        ?? throw new ArgumentException("Missing verified tools/list catalog.");
    var catalog = catalogDocument["tools"]?.AsArray() ?? throw new ArgumentException("Missing catalog tools.");
    var names = catalog.Select(t => t?["name"]?.GetValue<string>()
        ?? throw new ArgumentException("Catalog tool has no name.")).ToHashSet(StringComparer.Ordinal);
    var gateway = new ToolGateway(catalogDocument, authority: new OfflineAuthority());
    var rig = new RigContext("offline-rig", "offline-chat", false, DateTimeOffset.UtcNow,
        new Dictionary<int, (double, double)> { [2] = (0, 100) },
        new Dictionary<int, double> { [2] = 5 }, "offline-profile/session/state",
        new Dictionary<string, string> { ["camera"] = "synthetic-camera", ["switch"] = "synthetic-switch" },
        new Dictionary<int, SwitchCapability> { [2] = new("synthetic-channel-2", true, 0, 100, 5) });
    var exposed = gateway.ExposedTools(rig, proposals: true)["tools"]!.AsArray();
    var cases = new List<AuditCase>();
    var validationErrors = new List<CallError>();
    var checkedCalls = 0;
    var maxSerializedProposalCharacters = 0;
    foreach (var line in File.ReadLines(dataPath, Encoding.UTF8)) {
        if (string.IsNullOrWhiteSpace(line)) continue;
        var row = JsonNode.Parse(line)?.AsObject() ?? throw new ArgumentException("Dataset record must be an object.");
        if (row["provenance"]?["split"]?.GetValue<string>() != "train")
            throw new ArgumentException("Router development audit accepts train-only records.");
        var id = row["id"]?.GetValue<string>() ?? throw new ArgumentException("Missing example id.");
        var messages = row["messages"]?.AsArray() ?? throw new ArgumentException("Missing messages.");
        for (var turn = 0; turn < messages.Count; turn++) {
            if (messages[turn]?["tool_calls"] is not JsonArray calls) continue;
            foreach (var call in calls) {
                var function = call?["function"]?.AsObject() ?? throw new ArgumentException("Missing function call.");
                var tool = function["name"]!.GetValue<string>();
                var arguments = function["arguments"]!.AsObject();
                checkedCalls++;
                var proposal = new JsonObject { ["name"] = tool, ["arguments"] = arguments.DeepClone() };
                maxSerializedProposalCharacters = Math.Max(maxSerializedProposalCharacters, proposal.ToJsonString().Length);
                try {
                    gateway.Validate(tool, arguments);
                    var definition = exposed.FirstOrDefault(t => t?["name"]?.GetValue<string>() == tool)
                        ?? throw new ArgumentException("Tool absent from synthetic proposal catalog.");
                    var properties = definition["inputSchema"]!["properties"]!.AsObject();
                    if (arguments.Any(a => !properties.ContainsKey(a.Key)))
                        throw new ArgumentException("Argument is hidden by the proposal catalog.");
                }
                catch (ArgumentException ex) { validationErrors.Add(new(id, turn, tool, arguments.DeepClone(), ex.Message)); }
            }
        }
        if (row["category"]?.GetValue<string>() != "natural_chat_intent") continue;
        var locale = row["locale"]?.GetValue<string>() ?? throw new ArgumentException("Missing locale.");
        var expected = row["expected"]?["first_tool"]?.GetValue<string>()
            ?? throw new ArgumentException("Natural intent record has no expected first tool.");
        if (!names.Contains(expected)) throw new ArgumentException("Expected tool is absent from the verified catalog: " + expected);
        var message = messages.FirstOrDefault(m => m?["role"]?.GetValue<string>() == "user");
        var request = message?["content"]?.GetValue<string>() ?? throw new ArgumentException("Missing first user request.");
        var retrieved = ChatToolRouter.Select(request, exposed);
        cases.Add(new(id, locale, expected, request, retrieved, retrieved.Contains(expected, StringComparer.Ordinal)));
    }
    if (cases.Count == 0) throw new ArgumentException("No train-only natural_chat_intent examples found.");
    if (cases.Select(c => c.Id).Distinct(StringComparer.Ordinal).Count() != cases.Count)
        throw new ArgumentException("Duplicate natural intent ids.");

    static object Counts(IEnumerable<AuditCase> items) {
        var rows = items.ToArray();
        var found = rows.Count(r => r.ExpectedPresent);
        return new {
            cases = rows.Length,
            expectedToolRetrieved = found,
            missed = rows.Length - found,
            recall = (double)found / rows.Length
        };
    }

    static string Digest(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    var report = new {
        schemaVersion = "pins-chat-router-audit-1",
        generatedAt = DateTimeOffset.UtcNow,
        measurement = "Development training-intent retrieval recall; not unseen model accuracy.",
        hardwareExecution = false,
        modelInference = false,
        verifiedCatalogTools = names.Count,
        limitations = new[] {
            "Uses authored training requests, not held-out cases.",
            "Uses a synthetic fresh idle rig with a writable switch; live state may expose fewer tools.",
            "Measures whether retrieval supplies the expected first tool, not argument correctness or authorization.",
            "Argument checks use the real gateway validator, but cannot verify equipment state, approval or completion.",
            "Output token budgets require the pinned tokenizer; character counts are not token counts.",
            "Dataset changes cannot make a missing tool available to the model; production router changes require separate work."
        },
        inputs = new {
            trainSha256 = Digest(dataPath),
            catalogSha256 = Digest(catalogPath),
            runtimeAssemblySha256 = Digest(typeof(ChatToolRouter).Assembly.Location)
        },
        exposedTools = exposed,
        callValidation = new {
            callsChecked = checkedCalls,
            errors = validationErrors.Count,
            maxSerializedProposalCharacters,
            tokenBudgetMeasured = false,
            failures = validationErrors
        },
        summary = Counts(cases),
        byLocale = cases.GroupBy(c => c.Locale).OrderBy(g => g.Key, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => Counts(g), StringComparer.Ordinal),
        byTool = cases.GroupBy(c => c.ExpectedTool).OrderBy(g => g.Key, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => Counts(g), StringComparer.Ordinal),
        misses = cases.Where(c => !c.ExpectedPresent).Select(c => new {
            id = c.Id,
            locale = c.Locale,
            expectedTool = c.ExpectedTool,
            request = c.Request,
            retrieved = c.Retrieved
        }).ToArray()
    };
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n", new UTF8Encoding(false));
    Console.WriteLine(JsonSerializer.Serialize(new {
        summary = Counts(cases),
        callsChecked = checkedCalls,
        callErrors = validationErrors.Count,
        output = outputPath
    }));
    return validationErrors.Count == 0 ? 0 : 1;
}
catch (Exception ex) when (ex is ArgumentException or IOException or JsonException or InvalidOperationException) {
    Console.Error.WriteLine("Router audit failed: " + ex.Message);
    return 1;
}

internal sealed record AuditCase(string Id, string Locale, string ExpectedTool, string Request,
    string[] Retrieved, bool ExpectedPresent);
internal sealed record CallError(string Id, int AssistantTurn, string Tool, JsonNode Arguments, string Error);

// No approval, state refresh, network transport or hardware execution can occur.
internal sealed class OfflineAuthority : IOperatorAuthority {
    public Task<RigContext> RefreshAsync(string rigId, string chatId, CancellationToken ct) =>
        throw new InvalidOperationException("Offline audit must not refresh real equipment state.");
    public Task<OperatorApproval?> GetApprovalAsync(string requestId, CancellationToken ct) =>
        throw new InvalidOperationException("Offline audit must not obtain operator approval.");
}
