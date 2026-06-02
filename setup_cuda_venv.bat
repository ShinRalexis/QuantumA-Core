@echo off
setlocal

cd /d "%~dp0"
echo ==========================================
echo  QuantumA Core - CUDA venv setup
echo ==========================================
echo.

echo [1/5] Creating isolated venv .venv-cuda ...
python -m venv .venv-cuda
if errorlevel 1 goto :fail

echo [2/5] Activating venv ...
call .venv-cuda\Scripts\activate.bat
if errorlevel 1 goto :fail

echo [3/5] Upgrading pip ...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [4/5] Installing CUDA-enabled PyTorch (cu124) ...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :fail

echo [5/5] Verifying CUDA runtime ...
python -c "import torch; print('torch=', torch.__version__); print('cuda_version=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
if errorlevel 1 goto :fail

echo.
echo DONE. If cuda_available is True, you can run gpu_pro_validation.py in this venv.
exit /b 0

:fail
echo.
echo ERROR: CUDA venv setup failed.
exit /b 1
