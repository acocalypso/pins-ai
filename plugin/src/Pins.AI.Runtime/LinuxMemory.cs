namespace Pins.AI.Runtime;

public static class LinuxMemory {
    public static double? TemperatureC {
        get {
            try { return double.Parse(File.ReadAllText("/sys/class/thermal/thermal_zone0/temp"), System.Globalization.CultureInfo.InvariantCulture) / 1000; }
            catch (IOException) { return null; }
            catch (UnauthorizedAccessException) { return null; }
        }
    }
    public static MemorySample Read(int? pid = null) {
        if (!OperatingSystem.IsLinux()) return new(0, 0, 0);
        long ReadKiB(string path, string name) {
            try {
                var line = File.ReadLines(path).FirstOrDefault(x => x.StartsWith(name + ":", StringComparison.Ordinal));
                return line is null ? 0 : long.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1], System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (IOException) { return 0; }
            catch (UnauthorizedAccessException) { return 0; }
        }
        var available = ReadKiB("/proc/meminfo", "MemAvailable") / 1024;
        return pid is null ? new(available, 0, 0) : new(available,
            ReadKiB($"/proc/{pid}/status", "VmRSS") / 1024,
            ReadKiB($"/proc/{pid}/status", "VmSwap") / 1024);
    }

    public static bool WithinBudget(AiConfiguration config, MemorySample sample) =>
        sample.AvailableMiB >= config.MinAvailableMemoryMiB && sample.ResidentMiB <= config.MaxResidentMemoryMiB && sample.SwapMiB <= config.MaxProcessSwapMiB;
}
