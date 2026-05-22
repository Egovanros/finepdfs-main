# Run FinePDFs pipeline (UTF-8 console + venv)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:AWS_EC2_METADATA_DISABLED = "true"
# Common Crawl over HTTPS (default). Set to 1 only on AWS EC2 with credentials.
$env:FINEPDFS_CC_USE_S3 = "0"
# Windows CPU install has no vllm; skip RolmOCR / Qwen (steps 3 OCR branch, 4 OCR postprocess)
$env:FINEPDFS_SKIP_GPU_STEPS = "1"

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Run install.ps1 first."
    exit 1
}

.\.venv\Scripts\Activate.ps1
python run_finepdfs_pipeline.py @args
