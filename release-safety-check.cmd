@echo off
setlocal EnableExtensions

set "FAIL=0"

if exist "llm_models\blobs" (
  echo [FAIL] Found local model blobs in llm_models\blobs
  set "FAIL=1"
)
if exist "frontend\node_modules" (
  echo [FAIL] Found frontend\node_modules
  set "FAIL=1"
)
if exist "frontend\dist" (
  echo [FAIL] Found frontend\dist
  set "FAIL=1"
)
if exist ".venv" (
  echo [FAIL] Found .venv
  set "FAIL=1"
)
if exist "backend\.venv" (
  echo [FAIL] Found backend\.venv
  set "FAIL=1"
)

powershell -NoProfile -Command "$bad = Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '\.db($|-)' -or $_.Name -match '\.pyc($|\.)' }; if($bad){ $bad | Select-Object -First 20 -ExpandProperty FullName | %% { Write-Host ('[FAIL] forbidden file: ' + $_) }; exit 1 } else { exit 0 }"
if errorlevel 1 set "FAIL=1"

if "%FAIL%"=="1" (
  echo.
  echo Release safety check failed.
  exit /b 1
)

echo Release safety check passed.
exit /b 0
