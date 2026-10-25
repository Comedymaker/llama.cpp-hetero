$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$phoneRoot='/data/local/tmp/llama-exp-q23-no-mid-phone20-cpu8-20261025'
$targetRoot=Join-Path $taskRoot 'phone-results'
New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
# Collect only finalized cases; an in-flight log is not a final observation.
$completed=& adb shell "find $phoneRoot/status -name '*.completed'"
if($LASTEXITCODE -ne 0){throw 'Cannot read experiment status'}
$files=@('runners/run.log','prompts.sha256','route-input.sha256','new-library.sha256','logs/deployment-check.log','logs/weights-check.log','runner.pid')
foreach($marker in $completed){
 $caseName=([IO.Path]::GetFileName($marker.Trim()) -replace '\.completed$','')
 if($caseName -notmatch '^h\d{3}_(direct|staged)_t8_n128$'){continue}
 $files+=@("status/$caseName.completed","status/$caseName.exit","logs/$caseName.log","results/$caseName.txt","memory/$caseName/summary.txt")
}
foreach($fileName in $files){
 $target=Join-Path $targetRoot $fileName
 New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
 & adb pull "$phoneRoot/$fileName" $target
 if($LASTEXITCODE -ne 0){throw "Could not collect $fileName"}
}
foreach($optional in @('status/run.exit','status/completed','pairs.tsv')){
 & adb shell "test -e $phoneRoot/$optional"
 if($LASTEXITCODE -eq 0){
  $target=Join-Path $targetRoot $optional
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
  & adb pull "$phoneRoot/$optional" $target
  if($LASTEXITCODE -ne 0){throw "Could not collect $optional"}
 }
}
