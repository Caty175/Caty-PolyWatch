@echo off
REM Run all polymorphic demos to see their behavior
echo ============================================================
echo Running All Polymorphic Demonstrations
echo ============================================================
echo.

echo Press any key to run Demo 1 (Code Mutation)...
pause >nul
python poly_demo1.py
echo.
echo.

echo Press any key to run Demo 2 (Encryption/Decryption)...
pause >nul
python poly_demo2.py
echo.
echo.

echo Press any key to run Demo 3 (Code Obfuscation)...
pause >nul
python poly_demo3.py
echo.
echo.

echo Press any key to run Demo 4 (Self-Modification)...
pause >nul
python poly_demo4.py
echo.
echo.

echo Press any key to run PowerShell Demo...
pause >nul
powershell -ExecutionPolicy Bypass -File poly_demo.ps1
echo.
echo.

echo Press any key to run Batch Demo...
pause >nul
call poly_demo.bat
echo.
echo.

echo ============================================================
echo All Demonstrations Complete!
echo ============================================================
echo.
echo Key Observations:
echo - Each demo shows different polymorphic techniques
echo - Running the same demo multiple times produces different signatures
echo - All operations are completely benign
echo - Perfect for testing sandbox detection systems
echo.
pause

