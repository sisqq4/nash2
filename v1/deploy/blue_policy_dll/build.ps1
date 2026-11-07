[CmdletBinding()]
param(
    [string]$PythonExecutable = "D:\Anaconda\python.exe",
    [string]$BuildDirectory = "",
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$SourceDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $SourceDirectory "build"
}

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    $PythonCommand = Get-Command python -ErrorAction Stop
    $PythonExecutable = $PythonCommand.Source
}
$PythonRoot = Split-Path -Parent $PythonExecutable

$CMakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
if ($null -eq $CMakeCommand) {
    $BundledCMake = Join-Path ${env:ProgramFiles} `
        "Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (-not (Test-Path -LiteralPath $BundledCMake -PathType Leaf)) {
        throw "CMake was not found in PATH or the Visual Studio 2022 Community installation."
    }
    $CMakeExecutable = $BundledCMake
} else {
    $CMakeExecutable = $CMakeCommand.Source
}
$CTestExecutable = Join-Path (Split-Path -Parent $CMakeExecutable) "ctest.exe"

& $CMakeExecutable `
    -S $SourceDirectory `
    -B $BuildDirectory `
    -G "Visual Studio 17 2022" `
    -A x64 `
    "-DPython3_ROOT_DIR=$PythonRoot" `
    "-DPython3_EXECUTABLE=$PythonExecutable"
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }

& $CMakeExecutable --build $BuildDirectory --config Release --parallel 1
if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }

$PreviousPath = $env:PATH
try {
    $env:PATH = "$PythonRoot;$PythonRoot\Library\bin;$PreviousPath"
    & $CTestExecutable --test-dir $BuildDirectory -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "Native smoke test failed." }
} finally {
    $env:PATH = $PreviousPath
}

if ($Install) {
    $InstallDirectory = Join-Path $SourceDirectory "dist"
    & $CMakeExecutable --install $BuildDirectory --config Release --prefix $InstallDirectory
    if ($LASTEXITCODE -ne 0) { throw "CMake install failed." }
    Write-Host "Installed deployment bundle: $InstallDirectory"
}

Write-Host "Built DLL: $(Join-Path $BuildDirectory 'Release\blue_policy_bridge.dll')"
Write-Host "Built example: $(Join-Path $BuildDirectory 'Release\blue_policy_test_host.exe')"
