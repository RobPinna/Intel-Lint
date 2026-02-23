@echo off
setlocal EnableExtensions

REM Installs IPEX-LLM runtime for Ollama on Windows (Intel Arc/XPU path).
REM Usage:
REM   install-ipex-ollama.cmd

echo [1/4] Checking Python 3.11...
for /f "delims=" %%I in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PY311_EXE=%%I"
if not defined PY311_EXE (
  echo Python 3.11 not found. Installing with winget...
  winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements --disable-interactivity
  for /f "delims=" %%I in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PY311_EXE=%%I"
)
if not defined PY311_EXE (
  echo ERROR: Python 3.11 is required but could not be installed automatically.
  exit /b 1
)
echo Python 3.11: %PY311_EXE%

echo [2/4] Upgrading pip...
py -3.11 -m pip install --upgrade pip
if errorlevel 1 (
  echo ERROR: pip upgrade failed.
  exit /b 1
)

echo [3/4] Installing IPEX-LLM XPU runtime (xpu-2-6 profile)...
py -3.11 -m pip install --upgrade "ipex-llm[xpu-2-6]" --extra-index-url https://download.pytorch.org/whl/xpu
if errorlevel 1 (
  echo ERROR: IPEX-LLM XPU install failed.
  exit /b 1
)

echo [4/4] Verifying key packages...
py -3.11 -c "import torch, importlib.metadata as m; print('torch=', torch.__version__); print('ipex_llm=', m.version('ipex-llm'))"
if errorlevel 1 (
  echo ERROR: Runtime verification failed.
  exit /b 1
)

echo.
echo IPEX-LLM setup completed.
echo Next step:
echo   start-dev.cmd ollama ipex
exit /b 0
