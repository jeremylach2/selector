# Fine-tune Arm C: lyrics + metadata + measured audio features (combined)
# Runs: 2,068 train examples, 446 val examples, 3 epochs
# Expected duration: ~10-11 hours (longest prompts, combines A and B)
#
# Pass -ResumeFromCheckpoint <path> to continue an interrupted run instead of
# starting over, e.g.:
#   .\train_arm_c.ps1 -ResumeFromCheckpoint data\runs\C\checkpoint-4070

param(
  [string]$ResumeFromCheckpoint = ""
)

$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SplitsDir = Join-Path $ProjectRoot "data\splits"
$OutputDir = Join-Path $ProjectRoot "data\runs\C"
$HFHome = "D:\hf_cache"

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Set environment
$env:HF_HOME = $HFHome

Write-Host "=== Arm C Fine-Tune ===" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot"
Write-Host "Python: $VenvPython"
Write-Host "Splits Dir: $SplitsDir"
Write-Host "Output Dir: $OutputDir"
Write-Host "HF_HOME: $HFHome"
Write-Host ""

$StartTime = Get-Date
Write-Host "Start Time: $StartTime" -ForegroundColor Green

# Run training
if ($ResumeFromCheckpoint) {
  Write-Host "Resuming from checkpoint: $ResumeFromCheckpoint" -ForegroundColor Yellow
  & $VenvPython -m selector.tagger.train `
    --model qwen `
    --arm C `
    --splits-dir $SplitsDir `
    --output-dir $OutputDir `
    --epochs 3 `
    --resume-from-checkpoint $ResumeFromCheckpoint
} else {
  & $VenvPython -m selector.tagger.train `
    --model qwen `
    --arm C `
    --splits-dir $SplitsDir `
    --output-dir $OutputDir `
    --epochs 3
}

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "=== Complete ===" -ForegroundColor Green
Write-Host "End Time: $EndTime"
Write-Host "Total Duration: $($Duration.Hours)h $($Duration.Minutes)m $($Duration.Seconds)s"
Write-Host ""

# Show results summary
if (Test-Path (Join-Path $OutputDir "run_info.json")) {
  Write-Host "Results saved to: $OutputDir" -ForegroundColor Green
  Get-Content (Join-Path $OutputDir "run_info.json") | ConvertFrom-Json | Format-List
}
