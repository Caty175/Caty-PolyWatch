# Windows Sandbox Test Script
# Tests the Windows sandbox setup and functionality

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🧪 Testing Windows Sandbox Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Function to write success message
function Write-Success {
    param($Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

# Function to write error message
function Write-ErrorMessage {
    param($Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

# Function to write warning message
function Write-WarningMessage {
    param($Message)
    Write-Host "⚠️ $Message" -ForegroundColor Yellow
}

# Function to write info message
function Write-Info {
    param($Message)
    Write-Host "ℹ️ $Message" -ForegroundColor White
}

# Test 1: Check Python
Write-Host ""
Write-Info "Test 1: Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python found: $pythonVersion"
} catch {
    Write-ErrorMessage "Python not found!"
    Write-Info "Install from: https://www.python.org/downloads/"
    exit 1
}

# Test 2: Check psutil
Write-Host ""
Write-Info "Test 2: Checking psutil..."
$psutilCheck = python -c "import psutil; print('OK')" 2>&1
if ($psutilCheck -eq "OK") {
    Write-Success "psutil installed"
} else {
    Write-WarningMessage "psutil not installed"
    Write-Info "Install with: pip install psutil"
}

# Test 3: Check pywin32
Write-Host ""
Write-Info "Test 3: Checking pywin32..."
$pywin32Check = python -c "import win32api; print('OK')" 2>&1
if ($pywin32Check -eq "OK") {
    Write-Success "pywin32 installed"
} else {
    Write-WarningMessage "pywin32 not installed"
    Write-Info "Install with: pip install pywin32"
}

# Test 4: Check wmi
Write-Host ""
Write-Info "Test 4: Checking wmi..."
$wmiCheck = python -c "import wmi; print('OK')" 2>&1
if ($wmiCheck -eq "OK") {
    Write-Success "wmi installed"
} else {
    Write-WarningMessage "wmi not installed"
    Write-Info "Install with: pip install wmi"
}

# Test 5: Check if windows_sandbox.py exists
Write-Host ""
Write-Info "Test 5: Checking windows_sandbox.py..."
if (Test-Path "windows_sandbox.py") {
    Write-Success "windows_sandbox.py found"
} else {
    Write-ErrorMessage "windows_sandbox.py not found!"
    exit 1
}

# Test 6: Create test file
Write-Host ""
Write-Info "Test 6: Creating test file..."
$testScript = @"
@echo off
echo Test sample running
timeout /t 2 /nobreak > nul
echo Test complete
"@
$testScript | Out-File -FilePath "test_sample.bat" -Encoding ASCII
Write-Success "Test file created: test_sample.bat"

# Test 7: Run sandbox (basic)
Write-Host ""
Write-Info "Test 7: Running sandbox analysis (5 seconds)..."
try {
    $output = python windows_sandbox.py test_sample.bat --duration 5 2>&1
    if ($output -match "Analysis complete") {
        Write-Success "Sandbox analysis completed successfully"
    } else {
        Write-WarningMessage "Sandbox analysis may have issues"
        Write-Host $output
    }
} catch {
    Write-ErrorMessage "Sandbox analysis failed"
    Write-Host $_.Exception.Message
}

# Test 8: Check output files
Write-Host ""
Write-Info "Test 8: Checking output files..."
$reportFiles = Get-ChildItem -Path "logs" -Filter "*_report.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($reportFiles.Count -gt 0) {
    $latestReport = $reportFiles[0]
    Write-Success "Report file created: $($latestReport.Name)"
    
    # Validate JSON
    try {
        $json = Get-Content $latestReport.FullName | ConvertFrom-Json
        Write-Success "Report is valid JSON"
        
        # Show summary
        Write-Host ""
        Write-Info "Report Summary:"
        Write-Host "  • API Calls: $($json.summary.api_calls)" -ForegroundColor White
        Write-Host "  • Files Created: $($json.summary.files_created)" -ForegroundColor White
        Write-Host "  • Libraries Loaded: $($json.summary.libraries_loaded)" -ForegroundColor White
        Write-Host "  • Network Connections: $($json.summary.network_connections)" -ForegroundColor White
    } catch {
        Write-ErrorMessage "Report is not valid JSON"
    }
} else {
    Write-ErrorMessage "No report file found"
}

# Test 9: Test feature parser
Write-Host ""
Write-Info "Test 9: Testing feature parser..."
if (Test-Path "parse_behavioral_logs.py") {
    if ($reportFiles.Count -gt 0) {
        $latestReport = $reportFiles[0]
        try {
            $output = python parse_behavioral_logs.py --input $latestReport.FullName --output "test_features.csv" 2>&1
            if (Test-Path "test_features.csv") {
                Write-Success "Feature parser completed successfully"
                
                # Count features
                $csvContent = Get-Content "test_features.csv" -First 1
                $featureCount = ($csvContent -split ",").Count
                Write-Success "CSV created with $featureCount features"
            } else {
                Write-WarningMessage "Feature parser had issues"
            }
        } catch {
            Write-WarningMessage "Feature parser failed"
        }
    }
} else {
    Write-WarningMessage "parse_behavioral_logs.py not found"
}

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "📊 Test Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Info "Core Components:"
if (python --version 2>&1) { Write-Success "Python" } else { Write-ErrorMessage "Python" }
if ($psutilCheck -eq "OK") { Write-Success "psutil" } else { Write-WarningMessage "psutil" }
if ($pywin32Check -eq "OK") { Write-Success "pywin32" } else { Write-WarningMessage "pywin32" }
if ($wmiCheck -eq "OK") { Write-Success "wmi" } else { Write-WarningMessage "wmi" }

Write-Host ""
Write-Info "Functionality:"
if ($reportFiles.Count -gt 0) { Write-Success "Sandbox execution" } else { Write-ErrorMessage "Sandbox execution" }
if (Test-Path "test_features.csv") { Write-Success "Feature extraction" } else { Write-WarningMessage "Feature extraction" }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Success "Testing complete!"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Info "Next steps:"
Write-Host "  1. Install any missing components (see warnings above)" -ForegroundColor White
Write-Host "  2. Analyze malware: python windows_sandbox.py malware.exe --duration 120" -ForegroundColor White
Write-Host "  3. Convert features: python parse_behavioral_logs.py --input report.json --output features.csv" -ForegroundColor White
Write-Host "  4. Run prediction: python ..\Model\predict_lstm_behavioral.py --input features.csv" -ForegroundColor White
Write-Host "  5. Review documentation: WINDOWS_SANDBOX_GUIDE.md" -ForegroundColor White
Write-Host ""

Write-Info "Clean up test files:"
Write-Host "  Remove-Item test_sample.bat, test_features.csv" -ForegroundColor White
Write-Host ""
