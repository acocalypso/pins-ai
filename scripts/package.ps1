param(
    [Parameter(Mandatory=$true)][string]$HostDirectory,
    [string]$Dotnet = 'dotnet'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildOutput = Join-Path $repoRoot 'artifacts/plugin'
$packageRoot = Join-Path $repoRoot 'artifacts/package'
& $Dotnet build (Join-Path $repoRoot 'src/Pins.AI.Plugin/Pins.AI.Plugin.csproj') -c Release "-p:HostDirectory=$HostDirectory" -o $buildOutput
if ($LASTEXITCODE -ne 0) { throw 'Plugin build failed' }
New-Item -ItemType Directory -Force $packageRoot | Out-Null
foreach ($name in @('Pins.AI.Plugin.dll','Pins.AI.Runtime.dll')) {
    Copy-Item -LiteralPath (Join-Path $buildOutput $name) -Destination $packageRoot
}
foreach ($name in @('README.md','LICENSE','THIRD_PARTY_NOTICES.md')) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $name) -Destination $packageRoot
}
Copy-Item -LiteralPath (Join-Path $repoRoot 'native/pins-ai-exec.c') -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts/install-plugin.sh') -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $repoRoot 'models/manifest.json') -Destination (Join-Path $packageRoot 'model-manifest.json')
foreach ($folder in @('docs','models','native','benchmarks')) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $folder) -Destination $packageRoot -Recurse -Force
}
$scriptRoot = Join-Path $packageRoot 'scripts'
New-Item -ItemType Directory -Force $scriptRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts/install-assets.py') -Destination $scriptRoot
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts/benchmark.py') -Destination $scriptRoot
$hashes = Get-ChildItem -LiteralPath $packageRoot -File -Recurse | Where-Object Name -ne 'SHA256SUMS' | Sort-Object FullName | ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    $relativePath = $_.FullName.Substring($packageRoot.Length + 1).Replace('\','/')
    $hash.Hash.ToLowerInvariant() + '  ' + $relativePath
}
$hashes | Set-Content -LiteralPath (Join-Path $packageRoot 'SHA256SUMS') -Encoding ascii
$archive = Join-Path $repoRoot 'artifacts/pins-ai-0.1.0.zip'
Compress-Archive -Path (Join-Path $packageRoot '*') -DestinationPath $archive -Force
Write-Output "Package: $archive (models and native runtime installed separately, explicitly)"
