@echo off
setlocal
cd /d %~dp0
set MCP_SERVERS_CONFIG=configs\mcp_servers_production_agent.json
python scripts\main.py --mode both --env configs/envs/production_agent.env %*
if errorlevel 1 (
    echo.
    echo Production Agent exited with error level %errorlevel%.
)
echo.
pause
