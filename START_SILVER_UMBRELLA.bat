@echo off
color 0A
title Silver Umbrella - Starting...

echo.
echo ================================================================
echo           SILVER UMBRELLA - ONE-CLICK STARTUP
echo ================================================================
echo.

cd /d "%~dp0"

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Virtual environment not found!
    pause
    exit
)

echo [2/3] Starting Backend Server...
start "Silver Umbrella Backend" cmd /k "venv\Scripts\activate && python backend\app.py"
timeout /t 3 /nobreak >nul

echo [3/3] Starting Frontend Server...
start "Silver Umbrella Frontend" cmd /k "venv\Scripts\activate && cd frontend && python -m http.server 8000"
timeout /t 2 /nobreak >nul

echo.
echo ================================================================
echo              SILVER UMBRELLA IS NOW RUNNING!
echo ================================================================
echo.
echo   Backend:  http://localhost:5000
echo   Frontend: http://localhost:8000
echo.
echo   Opening browser in INCOGNITO mode (fresh cache)...
echo ================================================================
echo.

timeout /t 3 /nobreak >nul

REM Try Chrome incognito first
start chrome --incognito http://localhost:8000 2>nul
if errorlevel 1 (
    REM If Chrome fails, try Edge InPrivate
    start msedge --inprivate http://localhost:8000 2>nul
    if errorlevel 1 (
        REM If Edge fails, try Firefox private
        start firefox -private-window http://localhost:8000 2>nul
        if errorlevel 1 (
            REM If all fail, open normally
            start http://localhost:8000
        )
    )
)

echo.
echo Silver Umbrella is running!
echo Browser opened in private/incognito mode (no cache issues)
echo.
echo Close this window when done
echo.
pause
```

---

## SOLUTION 2: CLEAR BROWSER CACHE MANUALLY (ONE TIME)

### Chrome:
```
1. Press Ctrl + Shift + Delete
2. Select "Cached images and files"
3. Click "Clear data"
4. Close ALL Chrome windows
5. Run your .bat file again
```

### Edge:
```
1. Press Ctrl + Shift + Delete
2. Select "Cached images and files"
3. Click "Clear now"
4. Close ALL Edge windows
5. Run your .bat file again