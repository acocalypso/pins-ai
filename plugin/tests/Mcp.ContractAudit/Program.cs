using Microsoft.Extensions.DependencyInjection;
using NINA.Mcp;
using NINA.Mcp.Observatory;
using NINA.Mcp.Protocol;
using System.Reflection;
using System.Text.Json;

// Instantiate the team's existing dispatcher, never a second server or live hardware adapter.
var services = new ServiceCollection();
services.AddSingleton(DispatchProxy.Create<IPinsObservatory, NoHardware>());
services.AddPinsMcp(new McpServerOptions());
using var provider = services.BuildServiceProvider();
using var request = JsonDocument.Parse("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}");
var response = await provider.GetRequiredService<McpDispatcher>().DispatchAsync(request.RootElement, CancellationToken.None);
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[0]))!);
await File.WriteAllTextAsync(args[0], response.Response!["result"]!.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine("Exported actual tools/list; observatory operations forbidden.");

public class NoHardware : DispatchProxy {
    protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) => throw new InvalidOperationException("Audit forbids observatory calls.");
}
