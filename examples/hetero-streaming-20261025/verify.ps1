$ErrorActionPreference='Stop'
$package=$PSScriptRoot
$manifest=Import-Csv -LiteralPath "$package/files.sha256.csv"
foreach($row in $manifest){
 $path=Join-Path $package $row.RelativePath
 if(!(Test-Path -LiteralPath $path)){throw "Missing $($row.RelativePath)"}
 if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $row.SHA256){throw "Hash mismatch $($row.RelativePath)"}
}
"PASS $($manifest.Count) packaged files"
