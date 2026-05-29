# FinePDFs pipeline with Docling extraction (separate ./finepdfs/data_docling/ outputs)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:AWS_EC2_METADATA_DISABLED = "true"
$env:FINEPDFS_CC_USE_S3 = "0"
$env:FINEPDFS_SKIP_GPU_STEPS = "1"

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Run install.ps1 first."
    exit 1
}

.\.venv\Scripts\Activate.ps1
python run_finepdfs_pipeline_docling.py @args
