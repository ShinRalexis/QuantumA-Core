@echo off
setlocal
cd /d "%~dp0"
if not exist .venv-cuda\Scripts\python.exe (
  echo ERROR: .venv-cuda\Scripts\python.exe non trovato.
  exit /b 1
)

echo ===== CUDA VENv CHECK =====
.venv-cuda\Scripts\python.exe -c "import torch; print('torch_version=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
set RET=%ERRORLEVEL%
echo ===========================
exit /b %RET%
