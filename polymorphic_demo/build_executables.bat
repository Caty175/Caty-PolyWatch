@echo off
REM Helper script to build all Python demos into PE executables
REM Requires PyInstaller: pip install pyinstaller

echo ============================================================
echo Building Polymorphic Demo Executables
echo ============================================================

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller is not installed
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller
        echo [INFO] Please run: pip install pyinstaller
        pause
        exit /b 1
    )
)

echo [INFO] PyInstaller is installed
echo.

REM Create output directory
if not exist "executables" mkdir executables

REM Build each demo
echo [1/4] Building poly_demo1.exe...
pyinstaller --onefile --distpath executables --workpath build --specpath build poly_demo1.py
if errorlevel 1 (
    echo [ERROR] Failed to build poly_demo1.exe
) else (
    echo [SUCCESS] poly_demo1.exe created
)
echo.

echo [2/4] Building poly_demo2.exe...
pyinstaller --onefile --distpath executables --workpath build --specpath build poly_demo2.py
if errorlevel 1 (
    echo [ERROR] Failed to build poly_demo2.exe
) else (
    echo [SUCCESS] poly_demo2.exe created
)
echo.

echo [3/4] Building poly_demo3.exe...
pyinstaller --onefile --distpath executables --workpath build --specpath build poly_demo3.py
if errorlevel 1 (
    echo [ERROR] Failed to build poly_demo3.exe
) else (
    echo [SUCCESS] poly_demo3.exe created
)
echo.

echo [4/4] Building poly_demo4.exe...
pyinstaller --onefile --distpath executables --workpath build --specpath build poly_demo4.py
if errorlevel 1 (
    echo [ERROR] Failed to build poly_demo4.exe
) else (
    echo [SUCCESS] poly_demo4.exe created
)
echo.

REM Clean up build artifacts
echo [INFO] Cleaning up build artifacts...
if exist "build" rmdir /s /q build
if exist "*.spec" del /q *.spec

echo.
echo ============================================================
echo Build Complete!
echo ============================================================
echo.
echo Executables are located in: executables\
echo.
dir /b executables\*.exe
echo.
echo You now have 4 PE files ready for testing:
echo - poly_demo1.exe (Code Mutation)
echo - poly_demo2.exe (Encryption/Decryption)
echo - poly_demo3.exe (Code Obfuscation)
echo - poly_demo4.exe (Self-Modification)
echo.
echo Plus 2 script files:
echo - poly_demo.ps1 (PowerShell)
echo - poly_demo.bat (Batch)
echo.
echo ============================================================
pause

