$env:Path = "C:\msys64\mingw64\bin;" + $env:Path
Push-Location $PSScriptRoot
try {
	if ($args.Count -gt 0) {
		& "$PSScriptRoot\bin\image.exe" $args[0]
	} else {
		& "$PSScriptRoot\bin\image.exe"
	}
}
finally {
	Pop-Location
}