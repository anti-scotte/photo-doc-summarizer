@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Image Document Summarizer

echo ==============================================
echo  Image Document Summarizer
echo  Starting up, please wait...
echo ==============================================
echo.

if not exist "app\main.py" goto :nofiles

if exist "venv\Scripts\activate.bat" goto :activate

echo [1/4] Creating virtual environment...
py -3.12 -m venv venv >nul 2>&1
if not errorlevel 1 goto :checkver
echo       py -3.12 not available, trying python...
python -m venv venv
if errorlevel 1 goto :venvfail

:checkver
"venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11),(3,12)) else 1)" >nul 2>&1
if errorlevel 1 goto :badversion
echo       venv created.
goto :activate

:activate
echo [2/4] Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 goto :activatefail
python -c "import sys" >nul 2>&1
if errorlevel 1 goto :activatefail
echo       activated.

echo [3/4] Installing dependencies...
echo       this may take a few minutes on the first run
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto :pipfail
echo       dependencies ready.

echo [4/4] Starting server...
echo       local address: http://127.0.0.1:8000
echo       the browser opens automatically in 3 seconds
echo       keep this window open, press Ctrl+C to stop
echo.
start "" /min cmd /c "timeout /t 3 >nul 2>&1 || ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:8000"
python -m uvicorn app.main:app --port 8000

echo.
echo Server stopped.
pause
exit /b 0

:nofiles
color 0c
echo.
echo [ERROR] app\main.py not found.
echo         Put this file in the project root folder, next to
echo         the app folder, web folder and requirements.txt.
goto :end

:venvfail
color 0c
echo.
echo [ERROR] Could not create the virtual environment.
echo         Python 3.12 is required. Install it first:
echo         https://www.python.org/downloads/
echo         Then delete the venv folder and run this file again.
goto :end

:badversion
color 0c
echo.
echo [ERROR] Unsupported Python version in venv.
echo         Required: Python 3.10, 3.11 or 3.12
echo         Reason:   the OCR packages have no wheel for Python 3.13+
echo         Delete the venv folder, install Python 3.12, run again.
goto :end

:activatefail
color 0c
echo.
echo [ERROR] Could not activate the virtual environment.
echo         The venv folder looks broken.
echo         Delete the venv folder and run this file again.
goto :end

:pipfail
color 0c
echo.
echo [ERROR] pip install failed. Read the message above.
echo         1. Mirror unreachable? Open this file and remove
echo            -i https://pypi.tuna.tsinghua.edu.cn/simple
echo         2. Offline? Put the wheel files into a vendor folder.
echo         3. Missing VC++ runtime:
echo            https://aka.ms/vs/17/release/vc_redist.x64.exe
goto :end

:end
echo.
pause
exit /b 1