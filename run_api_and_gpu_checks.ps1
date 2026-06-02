$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path '.venv-cuda\Scripts\python.exe')) {
  throw '.venv-cuda\Scripts\python.exe non trovato.'
}

Write-Host '=== QuantumA Core API/GPU Check ==='
Write-Host '[1/3] GPU runtime check'
& '.venv-cuda\Scripts\python.exe' -c "import torch; print('torch_version=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

Write-Host '[2/3] 28 qubit smoke test'
& '.venv-cuda\Scripts\python.exe' .\test_28q_gpu.py

Write-Host '[3/3] API status reminder'
Write-Host 'Start API with: .venv-cuda\Scripts\python.exe api_server.py'
Write-Host 'Then test: GET /status, GET /status/gpu, POST /benchmark, POST /simulate'
Write-Host '===================================='
