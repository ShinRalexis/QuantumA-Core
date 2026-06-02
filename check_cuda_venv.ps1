$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$log = Join-Path $PSScriptRoot 'cuda_check.log'
if (-not (Test-Path '.venv-cuda\Scripts\python.exe')) {
  'ERROR: .venv-cuda\Scripts\python.exe non trovato.' | Set-Content -LiteralPath $log
  exit 1
}
& '.venv-cuda\Scripts\python.exe' -c "import torch; print('torch_version=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')" 2>&1 | Set-Content -LiteralPath $log
Write-Host "Log scritto in $log"
