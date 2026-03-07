$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$srcDir = Join-Path $root "benchmarks\native"
$buildDir = if ($env:BOOST_REF_BUILD_DIR) { $env:BOOST_REF_BUILD_DIR } else { Join-Path $root "stuff\migration\boost_ref_adapter\build_windows" }
$binPath = Join-Path $buildDir "Release\boost_ref_adapter.exe"
if (-not (Test-Path $binPath)) {
  $binPath = Join-Path $buildDir "boost_ref_adapter.exe"
}

$needsBuild = $true
if (Test-Path $binPath) {
  $binItem = Get-Item $binPath
  $cppItem = Get-Item (Join-Path $srcDir "boost_ref_adapter.cpp")
  $cmakeItem = Get-Item (Join-Path $srcDir "CMakeLists.txt")
  $needsBuild = ($cppItem.LastWriteTimeUtc -gt $binItem.LastWriteTimeUtc) -or ($cmakeItem.LastWriteTimeUtc -gt $binItem.LastWriteTimeUtc)
}

if ($needsBuild) {
  cmake -S $srcDir -B $buildDir -DCMAKE_BUILD_TYPE=Release | Out-Host
  cmake --build $buildDir --config Release | Out-Host
}

if (Test-Path (Join-Path $buildDir "Release\boost_ref_adapter.exe")) {
  $binPath = Join-Path $buildDir "Release\boost_ref_adapter.exe"
} else {
  $binPath = Join-Path $buildDir "boost_ref_adapter.exe"
}

& $binPath
