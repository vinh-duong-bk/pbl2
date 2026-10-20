$env:Path = "C:\msys64\mingw64\bin;" + $env:Path
Push-Location $PSScriptRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " PBL2: HE THONG NHAN DIEN KHUON MAT (VIOLA-JONES - SDL2)  " -ForegroundColor Green
Write-Host " Truong Dai hoc Bach Khoa - Dai hoc Da Nang              " -ForegroundColor Gray
Write-Host "==========================================================" -ForegroundColor Cyan

$exePath = "$PSScriptRoot\bin\gui_sdl.exe"

$sourceFiles = @(
    "sources/gui_sdl.cpp",
    "sources/face_detector.cpp",
    "sources/integral_image.cpp",
    "sources/wavelet.cpp",
    "sources/wlearner.cpp",
    "sources/vjlearner.cpp",
    "sources/model_io.cpp",
    "sources/pool.cpp"
)
$buildInputs = $sourceFiles + (Get-ChildItem (Join-Path $PSScriptRoot "headers") -Filter "*.h" -File | ForEach-Object {
    $_.FullName
})
$needsBuild = -not (Test-Path $exePath)
if (-not $needsBuild) {
    $exeTime = (Get-Item $exePath).LastWriteTimeUtc
    $needsBuild = $buildInputs | Where-Object {
        $inputPath = if ([System.IO.Path]::IsPathRooted($_)) { $_ } else { Join-Path $PSScriptRoot $_ }
        (Get-Item $inputPath).LastWriteTimeUtc -gt $exeTime
    } | Select-Object -First 1
    $needsBuild = [bool]$needsBuild
}

if ($needsBuild) {
    Write-Host "[Info] Dang bien dich lai gui_sdl.exe tu ma nguon hien tai..." -ForegroundColor Yellow

    $compiler = "C:\msys64\mingw64\bin\g++.exe"
    $includes = @(
        "-I", "headers",
        "-I", "C:\msys64\mingw64\include",
        "-I", "C:\msys64\mingw64\include\opencv5"
    )
    $libDirs = @("-L", "C:\msys64\mingw64\lib")
    $libs = @(
        "-lSDL2",
        "-lopencv_core", "-lopencv_imgproc",
        "-lopencv_imgcodecs", "-lopencv_xobjdetect", "-lopencv_videoio",
        "-lcomdlg32", "-lole32", "-lmf", "-lmfplat", "-lmfreadwrite", "-lmfuuid"
    )

    $buildCmd = @("-std=gnu++20", "-O3", "-g", "-static-libgcc", "-static-libstdc++") + $sourceFiles + $includes + $libDirs + $libs + @("-o", $exePath)

    Write-Host "Dang bien dich gui_sdl.exe (co toi uu hoa -O3 & SDL2)..." -ForegroundColor Yellow
    & $compiler $buildCmd

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[Loi] Bien dich that bai! Ma loi: $LASTEXITCODE" -ForegroundColor Red
        Pop-Location
        exit $LASTEXITCODE
    }
    Write-Host "[Thanh cong] Bien dich hoan tat!" -ForegroundColor Green
}

try {
    Write-Host "Khoi chay ung dung GUI SDL2 (kem console log truc tiep)..." -ForegroundColor Green
    & $exePath $args
}
finally {
    Pop-Location
}
