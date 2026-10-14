using System.Reflection;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using NINA.Mcp;
using NINA.Mcp.Observatory;
using NINA.Mcp.Protocol;

if (args.Length != 1) {
    throw new ArgumentException("Supply a new output JSON path. This tool only exports tools/list.");
}

var destination = Path.GetFullPath(args[0]);
if (File.Exists(destination)) {
    throw new IOException("Preserve the existing snapshot; use a new output path.");
}

var services = new ServiceCollection();
services.AddSingleton(DispatchProxy.Create<IPinsObservatory, NoEquipmentProxy>());
services.AddPinsMcp(new McpServerOptions { ReadOnly = false });
await using var provider = services.BuildServiceProvider();
var dispatcher = provider.GetRequiredService<McpDispatcher>();
using var request = JsonDocument.Parse("""{"jsonrpc":"2.0","id":1,"method":"tools/list"}""");
var response = await dispatcher.DispatchAsync(request.RootElement, CancellationToken.None);
var result = response.Response?["result"] ?? throw new InvalidOperationException("Dispatcher did not return tools/list.");
Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
await File.WriteAllTextAsync(destination, result.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"Exported {dispatcher.VisibleTools.Count} real dispatcher definitions; no equipment calls.");

public class NoEquipmentProxy : DispatchProxy {
    protected override object? Invoke(MethodInfo? method, object?[]? args) {
        throw new InvalidOperationException("Equipment access is prohibited during catalog export: " + method?.Name);
    }
}
