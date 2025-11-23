# Windows Sandbox Test Script
# Tests the Windows sandbox setup and functionality

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🧪 Testing Windows Sandbox Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Function to print success
function Print-Success {
    param($message)
    Write-Host "✅ $message" -ForegroundColor Green
}

# Function to print error
function Print-Error {
    param($message)
    Write-Host "❌ $message" -ForegroundColor Red
}

# Function to print warning
function Print-Warning {
    param($message)
    Write-Host "⚠️  $message" -ForegroundColor Yellow
}

# Function to print info
function Print-Info {
    param($message)
    Write-Host "ℹ️  $message" -ForegroundColor White
}

# --------------------------
# Test 1: Check Python
# --------------------------
Write-Host ""
Print-Info "Test 1: Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    Print-Success "Python found: $pythonVersion"
} catch {
    Print-Error "Python not found!"
    Print-Info "Install from: https://www.python.org/downloads/"
    exit 1
}

# --------------------------
# Test 2: Check psutil
# --------------------------
Write-Host ""
Print-Info "Test 2: Checking psutil..."
$psutilCheck = python -c "import psutil; print('OK')" 2>&1
if ($psutilCheck.Trim() -eq "OK") {
    Print-Success "psutil installed"
} else {
    Print-Warning "psutil not installed"
    Print-Info "Install with: pip install psutil"
}

# --------------------------
# Test 3: Check pywin32
# --------------------------
Write-Host ""
Print-Info "Test 3: Checking pywin32..."
$pywin32Check = python -c "import win32api; print('OK')" 2>&1
if ($pywin32Check.Trim() -eq "OK") {
    Print-Success "pywin32 installed"
} else {
    Print-Warning "pywin32 not installed"
    Print-Info "Install with: pip install pywin32"
}

# --------------------------
# Test 4: Check wmi
# --------------------------
Write-Host ""
Print-Info "Test 4: Checking wmi..."
$wmiCheck = python -c "import wmi; print('OK')" 2>&1
if ($wmiCheck.Trim() -eq "OK") {
    Print-Success "wmi installed"
} else {
    Print-Warning "wmi not installed"
    Print-Info "Install with: pip install wmi"
}

# --------------------------
# Test 5: Check windows_sandbox.py
# --------------------------
Write-Host ""
Print-Info "Test 5: Checking windows_sandbox.py..."
if (Test-Path "windows_sandbox.py") {
    Print-Success "windows_sandbox.py found"
} else {
    Print-Error "windows_sandbox.py not found!"
    exit 1
}

# --------------------------
# Test 6: Create test file
# --------------------------
Write-Host ""
Print-Info "Test 6: Creating test file..."
$testScript = @"
@echo off
echo Test sample running
timeout /t 2 /nobreak > nul
echo Test complete
"@
Set-Content -Path "test_sample.bat" -Value $testScript -Encoding ASCII
Print-Success "Test file created: test_sample.bat"

# --------------------------
# Test 7: Run sandbox (basic)
# --------------------------
Write-Host ""
Print-Info "Test 7: Running sandbox analysis (5 seconds)..."
try {
    $output = python windows_sandbox.py test_sample.bat --duration 5 2>&1
    if ($output -match "Analysis complete") {
        Print-Success "Sandbox analysis completed successfully"
    } else {
        Print-Warning "Sandbox analysis may have issues"
        Write-Host $output
    }
} catch {
    Print-Error "Sandbox analysis failed"
    Write-Host $_.Exception.Message
}

# --------------------------
# Test 8: Check output files
# --------------------------
Write-Host ""
Print-Info "Test 8: Checking output files..."
$reportFiles = Get-ChildItem -Path "logs" -Filter "*_report.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($reportFiles.Count -gt 0) {
    $latestReport = $reportFiles[0]
    Print-Success "Report file created: $($latestReport.Name)"
    
    # Validate JSON
    try {
        $json = Get-Content $latestReport.FullName | ConvertFrom-Json
        Print-Success "Report is valid JSON"
        
        # Show summary
        Write-Host ""
        Print-Info "Report Summary:"
        Write-Host "  • API Calls: $($json.summary.api_calls)" -ForegroundColor White
        Write-Host "  • Files Created: $($json.summary.files_created)" -ForegroundColor White
        Write-Host "  • Libraries Loaded: $($json.summary.libraries_loaded)" -ForegroundColor White
        Write-Host "  • Network Connections: $($json.summary.network_connections)" -ForegroundColor White
    } catch {
        Print-Error "Report is not valid JSON"
    }
} else {
    Print-Error "No report file found"
}

# --------------------------
# Test 9: Test feature parser
# --------------------------
Write-Host ""
Print-Info "Test 9: Testing feature parser..."
if (Test-Path "parse_behavioral_logs.py" -and $reportFiles.Count -gt 0) {
    $latestReport = $reportFiles[0]
    try {
        $output = python parse_behavioral_logs.py --input $latestReport.FullName --output "test_features.csv" 2>&1
        if (Test-Path "test_features.csv") {
            Print-Success "Feature parser completed successfully"
            
            # Count features
            $csvContent = Get-Content "test_features.csv" -First 1
            $featureCount = ($csvContent -split ",").Count
            Print-Success "CSV created with $featureCount features"
        } else {
            Print-Warning "Feature parser had issues"
        }
    } catch {
        Print-Warning "Feature parser failed"
    }
} else {
    Print-Warning "parse_behavioral_logs.py not found or no report file available"
}

# --------------------------
# Summary
# --------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "📊 Test Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Print-Info "Core Components:"
if (python --version 2>&1) { Print-Success "Python" } else { Print-Error "Python" }
if ($psutilCheck.Trim() -eq "OK") { Print-Success "psutil" } else { Print-Warning "psutil" }
if ($pywin32Check.Trim() -eq "OK") { Print-Success "pywin32" } else { Print-Warning "pywin32" }
if ($wmiCheck.Trim() -eq "OK") { Print-Success "wmi" } else { Print-Warning "wmi" }

Write-Host ""
Print-Info "Functionality:"
if ($reportFiles.Count -gt 0) { Print-Success "Sandbox execution" } else { Print-Error "Sandbox execution" }
if (Test-Path "test_features.csv") { Print-Success "Feature extraction" } else { Print-Warning "Feature extraction" }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Print-Success "Testing complete!"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Print-Info "Next steps:"
Write-Host "  1. Install any missing components (see warnings above)" -ForegroundColor White
Write-Host "  2. Analyze malware: python windows_sandbox.py malware.exe --duration 120" -ForegroundColor White
Write-Host "  3. Convert features: python parse_behavioral_logs.py --input report.json --output features.csv" -ForegroundColor White
Write-Host "  4. Run prediction: python ..\Model\predict_lstm_behavioral.py --input features.csv" -ForegroundColor White
Write-Host "  5. Review documentation: WINDOWS_SANDBOX_GUIDE.md" -ForegroundColor White
Write-Host ""

Print-Info "Clean up test files:"
Write-Host "  Remove-Item test_sample.bat, test_features.csv" -ForegroundColor White
Write-Host ""
