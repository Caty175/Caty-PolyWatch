@echo off
REM Polymorphic Demo - Batch Script
REM This demonstrates polymorphic behavior in batch files
REM EDUCATIONAL PURPOSE ONLY - No malicious behavior

echo ============================================================
echo POLYMORPHIC DEMONSTRATION - BATCH SCRIPT
echo Educational Purpose Only - No Malicious Behavior
echo ============================================================

REM Generate random mutation ID using time
set /a mutationId=%RANDOM% * 10 + 1000
echo [BAT DEMO] Mutation ID: %mutationId%

REM Generate random junk data
set junkData=
for /L %%i in (1,1,50) do call :AddJunkChar
echo [BAT DEMO] Junk Data Generated: %junkData:~0,20%...

REM Simulate different execution paths
set /a pathChoice=%RANDOM% %% 3
echo.
echo [BAT DEMO] Demonstrating Random Execution Paths
echo [BAT DEMO] Chosen Path: %pathChoice%

if %pathChoice%==0 goto PathA
if %pathChoice%==1 goto PathB
if %pathChoice%==2 goto PathC

:PathA
echo [BAT DEMO] Executing Path A
echo [BAT DEMO] Benign Operation: Current Date = %DATE%
goto ContinueExecution

:PathB
echo [BAT DEMO] Executing Path B
echo [BAT DEMO] Benign Operation: Current Time = %TIME%
goto ContinueExecution

:PathC
echo [BAT DEMO] Executing Path C
echo [BAT DEMO] Benign Operation: Computer Name = %COMPUTERNAME%
goto ContinueExecution

:ContinueExecution
echo.
echo [BAT DEMO] Simulating Code Obfuscation

REM Add junk operations
echo [BAT DEMO] Adding Junk Operations...
set /a junk1=%RANDOM% * %RANDOM%
set /a junk2=%RANDOM% + %RANDOM%
set /a junk3=%RANDOM% - 100
echo [BAT DEMO] Junk operations completed

REM Simulate encryption with simple XOR-like operation
echo.
echo [BAT DEMO] Demonstrating Simple Encoding
set "payload=BenignDemoPayload"
set /a encKey=%RANDOM% %% 255 + 1
echo [BAT DEMO] Original Payload: %payload%
echo [BAT DEMO] Encoding Key: %encKey%
echo [BAT DEMO] Encoded Payload: [Simulated - would be XOR encoded]

REM Demonstrate mutation
echo.
echo [BAT DEMO] Demonstrating Mutation Capability
set /a newMutationId=%RANDOM% * 10 + 1000
echo [BAT DEMO] Old Mutation ID: %mutationId%
echo [BAT DEMO] New Mutation ID: %newMutationId%
echo [BAT DEMO] IDs are different: TRUE

REM Show polymorphic characteristics
echo.
echo [BAT DEMO] Polymorphic Characteristics Demonstrated:
echo [BAT DEMO] 1. Random mutation IDs
echo [BAT DEMO] 2. Junk data insertion
echo [BAT DEMO] 3. Random execution paths
echo [BAT DEMO] 4. Code obfuscation simulation
echo [BAT DEMO] 5. Simple encoding demonstration
echo [BAT DEMO] 6. Self-mutation capability

echo.
echo [BAT DEMO] Demonstration Complete
echo ============================================================
goto :EOF

:AddJunkChar
set /a randChar=%RANDOM% %% 26 + 65
cmd /c exit /b %randChar%
set junkData=%junkData%%=ExitCode%
goto :EOF

