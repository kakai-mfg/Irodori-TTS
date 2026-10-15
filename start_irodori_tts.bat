@echo off
setlocal

title Irodori-TTS

set "IRODORI_DIR=%~dp0"
set "SERVER_HOST=127.0.0.1"
set "SERVER_PORT=7860"
set "WEB_URL=http://%SERVER_HOST%:%SERVER_PORT%"
set "UV_EXE="

echo.
echo ==========================================
echo   Starting Irodori-TTS
echo ==========================================
echo.

rem 1. Find uv from PATH.
for /f "delims=" %%I in ('where uv.exe 2^>nul') do (
    if not defined UV_EXE set "UV_EXE=%%I"
)

rem 2. Check the current official standalone installer location.
if not defined UV_EXE (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
    )
)

rem 3. Check the legacy installer location used by older uv versions.
if not defined UV_EXE (
    if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "UV_EXE=%USERPROFILE%\.cargo\bin\uv.exe"
    )
)

if not defined UV_EXE (
    echo ERROR: uv.exe was not found.
    echo.
    echo Install uv in PowerShell with:
    echo powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo.
    echo Then close and reopen this BAT file.
    echo.
    pause
    exit /b 1
)

if not exist "%IRODORI_DIR%gradio_app.py" (
    echo ERROR: gradio_app.py was not found.
    echo Put this BAT file in the Irodori-TTS folder.
    echo Current folder:
    echo %IRODORI_DIR%
    echo.
    pause
    exit /b 1
)

cd /d "%IRODORI_DIR%"
if errorlevel 1 (
    echo ERROR: Could not open the Irodori-TTS folder.
    pause
    exit /b 1
)

echo Opening a PowerShell window for the server.
echo Press Ctrl+C in that window to stop the server.
echo.

start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process '%WEB_URL%'"

rem Run the server in its own console so Ctrl+C does not interrupt a batch job.
start "Irodori-TTS" powershell.exe -NoLogo -NoProfile -Command "Write-Host ('Folder: ' + (Get-Location).Path); Write-Host ('uv: ' + $env:UV_EXE); Write-Host ('URL: ' + $env:WEB_URL); Write-Host ''; Write-Host 'Press Ctrl+C in this window to stop the server.'; Write-Host ''; & $env:UV_EXE run --no-sync python gradio_app.py --checkpoint Aratako/Irodori-TTS-v4-Large --model-device cuda --model-precision bf16 --codec-device cuda --codec-precision fp32 --server-name $env:SERVER_HOST --server-port $env:SERVER_PORT; exit $LASTEXITCODE"
exit /b %ERRORLEVEL%
