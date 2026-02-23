@echo off
setlocal EnableExtensions

REM Usage:
REM   start-dev.cmd
REM   start-dev.cmd ollama
REM   start-dev.cmd ollama vulkan
REM   start-dev.cmd ollama ipex
REM   start-dev.cmd ollama nvidia
REM   start-dev.cmd placeholder

set "ENGINE_MODE=%~1"
if "%ENGINE_MODE%"=="" set "ENGINE_MODE=placeholder"
set "OLLAMA_GPU_BACKEND_MODE=%~2"

set "ROOT_DIR=%~dp0"
set "BACKEND_DIR=%ROOT_DIR%backend"
set "FRONTEND_DIR=%ROOT_DIR%frontend"
set "OLLAMA_GPU_SCRIPT=%ROOT_DIR%start-ollama-gpu.cmd"
set "BACKEND_LAUNCHER=%TEMP%\intel_lint_backend.cmd"
set "FRONTEND_LAUNCHER=%TEMP%\intel_lint_frontend.cmd"
set "OLLAMA_LAUNCHER=%TEMP%\intel_lint_ollama.cmd"
set "PYTHON_EXE="
set "PYTHON_PRIMARY="

if /I not "%ENGINE_MODE%"=="placeholder" if /I not "%ENGINE_MODE%"=="ollama" (
  echo Invalid engine: %ENGINE_MODE%
  echo Valid values: placeholder or ollama
  exit /b 1
)
if /I "%ENGINE_MODE%"=="ollama" (
  if "%OLLAMA_GPU_BACKEND_MODE%"=="" (
    set "OLLAMA_GPU_BACKEND_MODE=ipex"
  )
)
if /I "%ENGINE_MODE%"=="ollama" (
  if /I not "%OLLAMA_GPU_BACKEND_MODE%"=="vulkan" if /I not "%OLLAMA_GPU_BACKEND_MODE%"=="ipex" if /I not "%OLLAMA_GPU_BACKEND_MODE%"=="nvidia" (
    echo Invalid Ollama GPU backend: %OLLAMA_GPU_BACKEND_MODE%
    echo Valid values: vulkan, ipex, or nvidia
    exit /b 1
  )
)

where py >nul 2>nul
if not errorlevel 1 (
  for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
)
for /f "delims=" %%I in ('where python 2^>nul') do (
  if not defined PYTHON_PRIMARY set "PYTHON_PRIMARY=%%I"
  echo %%I | findstr /I /C:"\\WindowsApps\\" >nul
  if errorlevel 1 if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
)
if not defined PYTHON_EXE set "PYTHON_EXE=%PYTHON_PRIMARY%"
if not defined PYTHON_EXE (
  echo Could not resolve a valid Python executable path.
  echo Install Python and ensure either "py -3" or "python" works in PATH.
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo npm not found in PATH.
  exit /b 1
)

if not exist "%BACKEND_DIR%\requirements.txt" (
  echo Backend folder not found: "%BACKEND_DIR%"
  exit /b 1
)

if not exist "%FRONTEND_DIR%\package.json" (
  echo Frontend folder not found: "%FRONTEND_DIR%"
  exit /b 1
)

(
  echo @echo off
  echo set "PYTHON_EXE=%PYTHON_EXE%"
  echo set "VENV_PY=.venv\Scripts\python.exe"
  echo cd /d "%BACKEND_DIR%"
  echo if not exist "%%VENV_PY%%" "%%PYTHON_EXE%%" -m venv .venv
  echo if not exist "%%VENV_PY%%" ^(
  echo   echo ERROR: venv python not found at %%VENV_PY%%
  echo   echo Python used: %%PYTHON_EXE%%
  echo   pause
  echo   exit /b 1
  echo ^)
  echo "%%VENV_PY%%" -m pip install -q -r requirements.txt
  echo if errorlevel 1 ^(
  echo   echo ERROR: pip install failed in backend.
  echo   pause
  echo   exit /b 1
  echo ^)
  echo set ENGINE=%ENGINE_MODE%
  if /I "%ENGINE_MODE%"=="ollama" (
    echo set OLLAMA_HOST=http://127.0.0.1:11434
    echo set OLLAMA_HOME=%ROOT_DIR%llm_models
    echo set OLLAMA_MODELS=%ROOT_DIR%llm_models
    echo set OLLAMA_MODEL=foundation-sec:latest
    echo set no_proxy=localhost,127.0.0.1,::1
    echo set NO_PROXY=localhost,127.0.0.1,::1
    echo set OLLAMA_REQUIRE_GPU=0
    echo set OLLAMA_NUM_GPU=999
    echo set OLLAMA_CPU_FALLBACK_ON_GPU_FAILURE=0
    echo set OLLAMA_GPU_SAFE_MODE_ON_ERROR=1
    echo set OLLAMA_GPU_SAFE_NUM_CTX=2048
    echo set OLLAMA_GPU_SAFE_NUM_BATCH=64
    echo set OLLAMA_GPU_SAFE_MAX_CHARS_PER_CHUNK=3500
    echo set OLLAMA_NUM_PREDICT=800
    echo set OLLAMA_NUM_CTX=3072
    echo set OLLAMA_NUM_BATCH=128
    echo set OLLAMA_SEED=42
    echo set OLLAMA_DETERMINISTIC_CACHE=1
    echo set OLLAMA_DETERMINISTIC_CACHE_MAX_ENTRIES=256
    echo set OLLAMA_STRICT_DETERMINISM=1
    echo set OLLAMA_STRICT_DETERMINISM_ATTEMPTS=2
    echo set OLLAMA_MAX_CALLS=6
    echo set OLLAMA_MAX_SPLIT_DEPTH=1
    echo set OLLAMA_MAX_CLAIMS=4
    echo set OLLAMA_MAX_CHARS_PER_CHUNK=5000
    echo set OLLAMA_CHUNK_OVERLAP_CHARS=40
  )
  echo "%%VENV_PY%%" -m uvicorn app.main:app --reload --port 8000
) > "%BACKEND_LAUNCHER%"

(
  echo @echo off
  echo cd /d "%FRONTEND_DIR%"
  echo if not exist "node_modules" npm install
  echo npm run dev -- --host 127.0.0.1 --port 5173
) > "%FRONTEND_LAUNCHER%"

(
  echo @echo off
  echo cd /d "%ROOT_DIR%"
  if /I "%ENGINE_MODE%"=="ollama" (
    echo set OLLAMA_GPU_BACKEND=%OLLAMA_GPU_BACKEND_MODE%
  )
  echo call "%OLLAMA_GPU_SCRIPT%"
) > "%OLLAMA_LAUNCHER%"

if /I "%ENGINE_MODE%"=="ollama" (
  if exist "%OLLAMA_GPU_SCRIPT%" (
    echo Starting Ollama GPU service ^(backend=%OLLAMA_GPU_BACKEND_MODE%^)...
    start "Intel Lint Ollama GPU" cmd /k "call ""%OLLAMA_LAUNCHER%"""
    timeout /t 4 >nul
  )
)

if /I "%ENGINE_MODE%"=="ollama" (
  powershell -NoProfile -Command "$ok=$false; 1..20 | ForEach-Object { try { Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null; $ok=$true; break } catch { Start-Sleep -Seconds 1 } }; if($ok){exit 0}else{exit 1}" >nul 2>nul
  if errorlevel 1 (
    echo WARNING: Ollama is not reachable at http://127.0.0.1:11434
    echo          Backend may return connection errors until Ollama is up.
  ) else (
    echo Ollama is reachable at http://127.0.0.1:11434
  )
)

echo Starting backend (ENGINE=%ENGINE_MODE%)...
start "Intel Lint Backend" cmd /k "call ""%BACKEND_LAUNCHER%"""

echo Starting frontend...
start "Intel Lint Frontend" cmd /k "call ""%FRONTEND_LAUNCHER%"""

echo Done. Backend: http://127.0.0.1:8000  Frontend: http://127.0.0.1:5173
exit /b 0
