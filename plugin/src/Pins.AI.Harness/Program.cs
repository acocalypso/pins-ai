using System.Security.Cryptography;
using Pins.AI.Runtime;

// Development-only lifecycle harness. Do not install or run as an AI service.
if (args.Length != 2) {
    Console.Error.WriteLine("usage: Pins.AI.Harness config.json data-directory");
    return 64;
}
var config = await AiConfiguration.ReadAsync(args[0]);
Directory.CreateDirectory(args[1]);
var keyPath = Path.Combine(args[1], "api.key");
if (!File.Exists(keyPath)) LlamaRuntime.WritePrivateFile(keyPath, Convert.ToHexString(RandomNumberGenerator.GetBytes(32)));
await using var runtime = new LlamaRuntime(config, args[1]);
await using var api = new LocalApi(runtime, args[0], (await File.ReadAllTextAsync(keyPath)).Trim());
using var shutdown = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; shutdown.Cancel(); };
using var sigterm = System.Runtime.InteropServices.PosixSignalRegistration.Create(
    System.Runtime.InteropServices.PosixSignal.SIGTERM, context => { context.Cancel = true; shutdown.Cancel(); });
api.Start();
await runtime.InitializeAsync();
Console.WriteLine("Development harness ready; no MCP or equipment tools.");
try { await Task.Delay(Timeout.InfiniteTimeSpan, shutdown.Token); } catch (OperationCanceledException) { }
return 0;
