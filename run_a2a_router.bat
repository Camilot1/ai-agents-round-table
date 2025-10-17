@echo off
setlocal
cd /d %~dp0
set ROUTER_CONFIG=configs\a2a_agents.json
python scripts\a2a_router.py --config "%ROUTER_CONFIG%" --host 127.0.0.1 --port 7080 %*
if errorlevel 1 (
    echo.
    echo A2A router exited with error level %errorlevel%.
)
echo.
pause
