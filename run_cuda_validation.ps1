$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path '.venv-cuda\Scripts\python.exe')) {
  throw '.venv-cuda\Scripts\python.exe non trovato. Esegui prima setup_cuda_venv.bat.'
}

Write-Host '[1/3] Installazione PyTorch CUDA nella venv isolata...'
& '.venv-cuda\Scripts\python.exe' -m pip install --upgrade pip
& '.venv-cuda\Scripts\python.exe' -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host '[2/3] Verifica runtime CUDA...'
& '.venv-cuda\Scripts\python.exe' -c "import torch; print('torch_version=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

Write-Host '[3/3] Test GPU del progetto...'
& '.venv-cuda\Scripts\python.exe' .\gpu_pro_validation.py
