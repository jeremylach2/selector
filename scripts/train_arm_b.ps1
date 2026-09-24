# Fine-tune Arm B: measured audio features only
# Runs: 2,068 train examples, 446 val examples, 3 epochs
# Expected duration: ~7-8 hours (shorter prompts than Arm A)

$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SplitsDir = Join-Path $ProjectRoot "data\splits"
$OutputDir = Join-Path $ProjectRoot "data\runs\B"
$HFHome = "D:\hf_cache"

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Set environment
$env:HF_HOME = $HFHome

Write-Host "=== Arm B Fine-Tune ===" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot"
Write-Host "Python: $VenvPython"
Write-Host "Splits Dir: $SplitsDir"
Write-Host "Output Dir: $OutputDir"
Write-Host "HF_HOME: $HFHome"
Write-Host ""

$StartTime = Get-Date
Write-Host "Start Time: $StartTime" -ForegroundColor Green

# Run training
& $VenvPython -m selector.tagger.train `
  --model qwen `
  --arm B `
  --splits-dir $SplitsDir `
  --output-dir $OutputDir `
  --epochs 3

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
