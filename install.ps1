# FinePDFs install script for Windows (CPU / no vLLM)
# Usage: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# MSVC flags required to build fasttext-numpy2-wheel on Windows
$env:CL = "/std:c++17 /Dssize_t=intptr_t"
$env:CXXFLAGS = "/std:c++17"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    py -3.12 -m pip install uv
}

if (-not (Test-Path ".venv")) {
    uv venv -p 3.12
}

Write-Host "Syncing dependencies (without GPU packages vllm/flash-attn)..."
uv sync --no-build-isolation-package fasttext-numpy2-wheel

Write-Host ""
Write-Host "Installing Windows libmagic DLL (python-magic-bin)..."
uv pip install python-magic-bin

Write-Host "Verifying imports..."
.\.venv\Scripts\python.exe -c "import magic; import fasttext; import datatrove; import opendataloader_pdf; print('magic + fasttext + datatrove + opendataloader-pdf OK')"

$javaCheck = .\.venv\Scripts\python.exe -c "from blocks.extractors.opendataloader import java_version_ok; ok, msg = java_version_ok(); print(msg); raise SystemExit(0 if ok else 1)" 2>&1
$javaExit = $LASTEXITCODE
$javaCheck | ForEach-Object { Write-Host $_ }
if ($javaExit -ne 0) {
    Write-Host "WARNING: Step 3 (OpenDataLoader PDF) will be skipped until Java 11+ is installed."
    Write-Host "  Install Temurin 17: https://adoptium.net/"
    Write-Host "  If java -version still shows 1.8, set JAVA_HOME to jdk-17 and put %JAVA_HOME%\bin before Oracle java8path in PATH."
}

Write-Host ""
Write-Host "Done. Activate and run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  .\run.ps1 --crawl-ids CC-MAIN-2024-10 --gpus 0"
Write-Host ""
Write-Host "PDF text extraction: OpenDataLoader PDF (requires Java). RolmOCR/Qwen need Linux + NVIDIA GPU."
