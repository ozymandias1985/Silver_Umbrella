@echo off
title Silver Umbrella Startup

echo ========================================
echo    SILVER UMBRELLA - STARTING
echo ========================================
echo.

cd /d "%~dp0"

echo Starting backend...
start "Silver Umbrella - Backend" cmd /k "venv\Scripts\activate && python backend\app.py"

timeout /t 3 /nobreak >nul

echo Starting frontend...
start "Silver Umbrella - Frontend" cmd /k "cd frontend && python -m http.server 8000"

timeout /t 2 /nobreak >nul

echo Opening browser...
start http://localhost:8000

echo.
echo ========================================
echo  SILVER UMBRELLA IS RUNNING!
echo  Frontend: http://localhost:8000
echo  Backend: http://localhost:5000
echo ========================================
echo.
echo Press any key to close this window
echo (DO NOT close the other 2 windows!)
pause >nul