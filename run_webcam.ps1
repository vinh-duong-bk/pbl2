$env:Path = "C:\msys64\mingw64\bin;" + $env:Path
Push-Location $PSScriptRoot
try {
	& "$PSScriptRoot\bin\webcam.exe"
}
finally {
	Pop-Location
}