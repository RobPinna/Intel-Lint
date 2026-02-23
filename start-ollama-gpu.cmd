@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Starts Ollama in IPEX-LLM XPU mode (Intel), NVIDIA CUDA mode, or Vulkan.
REM Usage:
REM   start-ollama-gpu.cmd
REM   start-ollama-gpu.cmd ipex
REM   start-ollama-gpu.cmd vulkan
REM   start-ollama-gpu.cmd nvidia

set "OLLAMA_EXE="
for /f "delims=" %%I in ('where.exe ollama.exe 2^>nul') do if not defined OLLAMA_EXE set "OLLAMA_EXE=%%I"
if not defined OLLAMA_EXE (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
)
if not defined OLLAMA_EXE (
  echo Ollama not found in PATH.
  exit /b 1
)

set "no_proxy=localhost,127.0.0.1"
set "NO_PROXY=localhost,127.0.0.1"
set "OLLAMA_HOST=http://127.0.0.1:11434"
if not defined OLLAMA_HOME set "OLLAMA_HOME=%~dp0llm_models"
if not defined OLLAMA_MODELS set "OLLAMA_MODELS=%~dp0llm_models"

if not "%~1"=="" set "OLLAMA_GPU_BACKEND=%~1"
set "OLLAMA_GPU_BACKEND=%OLLAMA_GPU_BACKEND%"
if "%OLLAMA_GPU_BACKEND%"=="" set "OLLAMA_GPU_BACKEND=ipex"
if /I not "%OLLAMA_GPU_BACKEND%"=="ipex" if /I not "%OLLAMA_GPU_BACKEND%"=="vulkan" if /I not "%OLLAMA_GPU_BACKEND%"=="nvidia" (
  echo Invalid OLLAMA_GPU_BACKEND=%OLLAMA_GPU_BACKEND%
  echo Valid values: ipex, vulkan, or nvidia
  exit /b 1
)

if /I "%OLLAMA_GPU_BACKEND%"=="ipex" (
  set "PY311_EXE="
  set "PY311_DIR="
  set "IPEX_LIB_DIR="
  for /f "delims=" %%I in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do if not defined PY311_EXE set "PY311_EXE=%%I"
  if not defined PY311_EXE (
    echo Python 3.11 was not found.
    echo Run install-ipex-ollama.cmd first.
    exit /b 1
  )
  for %%A in ("!PY311_EXE!") do set "PY311_DIR=%%~dpA"
  if "!PY311_DIR:~-1!"=="\" set "PY311_DIR=!PY311_DIR:~0,-1!"
  set "PATH=!PY311_DIR!;!PY311_DIR!\Scripts;!PY311_DIR!\Library\bin;!PATH!"

  for /f "delims=" %%I in ('py -3.11 -c "import bigdl.cpp, pathlib; print(pathlib.Path(bigdl.cpp.__file__).resolve().parent/\"libs\")" 2^>nul') do if not defined IPEX_LIB_DIR set "IPEX_LIB_DIR=%%I"
  if not defined IPEX_LIB_DIR (
    echo Could not resolve BigDL IPEX runtime directory.
    echo Ensure ipex-llm[xpu-2-6] is installed in Python 3.11.
    exit /b 1
  )
  if not exist "!IPEX_LIB_DIR!\ollama.exe" (
    echo IPEX runtime is missing ollama.exe in "!IPEX_LIB_DIR!".
    exit /b 1
  )

  set "IPEX_RUNTIME_DIR=%LOCALAPPDATA%\CTIClaimsGuard\ollama-ipex-runtime"
  if not exist "!IPEX_RUNTIME_DIR!" mkdir "!IPEX_RUNTIME_DIR!"

  call :copy_runtime_file "ollama.exe"
  if errorlevel 1 exit /b 1
  call :copy_runtime_file "ollama-lib.exe"
  if errorlevel 1 exit /b 1
  call :copy_runtime_file "ollama_llama.dll"
  if errorlevel 1 exit /b 1
  call :copy_runtime_file "ollama_ggml.dll"
  if errorlevel 1 exit /b 1
  call :copy_runtime_file "ollama_llava_shared.dll"
  if errorlevel 1 exit /b 1
  call :copy_runtime_file "ollama-ggml-base.dll"
  if errorlevel 1 exit /b 1
  call :copy_runtime_file "ollama-ggml-cpu.dll"
  if errorlevel 1 exit /b 1
  call :copy_runtime_file "ollama-ggml-sycl.dll"
  if errorlevel 1 exit /b 1
  call :copy_runtime_file "libc++.dll"
  if errorlevel 1 exit /b 1

  set "OLLAMA_EXE=!IPEX_RUNTIME_DIR!\ollama.exe"
  set "PATH=!IPEX_RUNTIME_DIR!;!PATH!"

  REM Prefer XPU path and disable Vulkan-specific overrides.
  set "OLLAMA_VULKAN=false"
  set "GGML_VK_VISIBLE_DEVICES="
  set "OLLAMA_LLM_LIBRARY=oneapi"
  set "SYCL_CACHE_PERSISTENT=1"
  set "BIGDL_LLM_XMX_DISABLED=0"
  set "ONEAPI_DEVICE_SELECTOR=level_zero:gpu"
  set "OLLAMA_INTEL_GPU=1"
) else if /I "%OLLAMA_GPU_BACKEND%"=="vulkan" (
  REM Prefer Vulkan backend for Intel Arc if oneAPI/IPEX is unavailable.
  set "OLLAMA_VULKAN=1"
  set "OLLAMA_LLM_LIBRARY="
  set "ONEAPI_DEVICE_SELECTOR="
  set "OLLAMA_INTEL_GPU="
  set "SYCL_CACHE_PERSISTENT="
  set "BIGDL_LLM_XMX_DISABLED="
) else (
  REM NVIDIA mode: use default Ollama runtime and CUDA autodetection.
  set "OLLAMA_VULKAN=false"
  set "OLLAMA_LLM_LIBRARY="
  set "ONEAPI_DEVICE_SELECTOR="
  set "OLLAMA_INTEL_GPU="
  set "GGML_VK_VISIBLE_DEVICES="
  set "SYCL_CACHE_PERSISTENT="
  set "BIGDL_LLM_XMX_DISABLED="
)

REM Full layer offload hint where supported by backend.
set "OLLAMA_NUM_GPU=999"
set "OLLAMA_NUM_PARALLEL=1"
set "OLLAMA_MAX_LOADED_MODELS=1"
echo Ollama executable: %OLLAMA_EXE%

echo Stopping existing Ollama processes...
taskkill /F /IM ollama.exe >nul 2>nul
powershell -NoProfile -Command "Get-Process | Where-Object { $_.ProcessName -like 'ollama*' } | Stop-Process -Force -ErrorAction SilentlyContinue" >nul 2>nul

echo Starting Ollama with:
echo   OLLAMA_GPU_BACKEND=%OLLAMA_GPU_BACKEND%
if defined PY311_EXE echo   PY311_EXE=%PY311_EXE%
if defined IPEX_LIB_DIR echo   IPEX_LIB_DIR=%IPEX_LIB_DIR%
if defined IPEX_RUNTIME_DIR echo   IPEX_RUNTIME_DIR=%IPEX_RUNTIME_DIR%
echo   OLLAMA_NUM_GPU=%OLLAMA_NUM_GPU%
echo   OLLAMA_NUM_PARALLEL=%OLLAMA_NUM_PARALLEL%
echo   OLLAMA_MAX_LOADED_MODELS=%OLLAMA_MAX_LOADED_MODELS%
echo   OLLAMA_VULKAN=%OLLAMA_VULKAN%
if defined OLLAMA_LLM_LIBRARY echo   OLLAMA_LLM_LIBRARY=%OLLAMA_LLM_LIBRARY%
if defined ONEAPI_DEVICE_SELECTOR echo   ONEAPI_DEVICE_SELECTOR=%ONEAPI_DEVICE_SELECTOR%
if defined OLLAMA_INTEL_GPU echo   OLLAMA_INTEL_GPU=%OLLAMA_INTEL_GPU%
echo   OLLAMA_EXE=%OLLAMA_EXE%
echo.
echo Keep this window open while backend is running.
"%OLLAMA_EXE%" serve
exit /b %errorlevel%

:copy_runtime_file
set "RUNTIME_FILE=%~1"
if not exist "%IPEX_LIB_DIR%\%RUNTIME_FILE%" (
  echo Missing IPEX runtime file: %IPEX_LIB_DIR%\%RUNTIME_FILE%
  exit /b 1
)
copy /Y "%IPEX_LIB_DIR%\%RUNTIME_FILE%" "%IPEX_RUNTIME_DIR%\%RUNTIME_FILE%" >nul
if errorlevel 1 (
  echo Failed to copy runtime file: %RUNTIME_FILE%
  exit /b 1
)
exit /b 0
