# 🪟 PowerShell Quick Start Guide

**For Windows Host Users**

---

## ⚡ **Quick Commands (Copy & Paste)**

### **1. Test Connection to VM**

```powershell
python windows_sandbox_client.py --vm-ip 192.168.100.33 --health
```

### **2. Analyze a File**

```powershell
python windows_sandbox_client.py --vm-ip 192.168.100.33 --file malware.exe --duration 120
```

### **3. List All Reports**

```powershell
python windows_sandbox_client.py --vm-ip 192.168.100.33 --list-reports
```

### **4. Download Specific Report**

```powershell
python windows_sandbox_client.py --vm-ip 192.168.100.33 --get-report <analysis_id>
```

### **5. Convert to LSTM Format**

```powershell
python parse_behavioral_logs.py --input <analysis_id>_report.json --output features.csv
```

### **6. Run LSTM Prediction**

```powershell
python ..\Model\predict_lstm_behavioral.py --input features.csv
```

---

## 📋 **Complete Workflow (PowerShell)**

```powershell
# Step 1: Test connection
python windows_sandbox_client.py --vm-ip 192.168.100.33 --health

# Step 2: Analyze malware
python windows_sandbox_client.py --vm-ip 192.168.100.33 --file malware.exe --duration 120

# Step 3: Note the analysis_id from output (e.g., abc123-def456-...)

# Step 4: Convert to LSTM format
python parse_behavioral_logs.py --input abc123-def456-..._report.json --output features.csv

# Step 5: Run prediction
python ..\Model\predict_lstm_behavioral.py --input features.csv
```

---

## 🔧 **PowerShell vs Bash Differences**

| Feature | PowerShell (Windows) | Bash (Linux/Mac) |
|---------|---------------------|------------------|
| **Line continuation** | Backtick `` ` `` | Backslash `\` |
| **Path separator** | Backslash `\` | Forward slash `/` |
| **Parent directory** | `..` | `..` |
| **Example multi-line** | `command `` <br>  --arg value` | `command \`<br>  --arg value` |

---

## 💡 **PowerShell Multi-line Commands**

If you want to split long commands across multiple lines in PowerShell:

```powershell
# Use backtick ` at the end of each line (NOT backslash \)
python windows_sandbox_client.py `
  --vm-ip 192.168.100.33 `
  --file malware.exe `
  --duration 120
```

**Important:** 
- Use **backtick `` ` ``** (above Tab key)
- NOT backslash `\` (that's for Bash)
- No space after the backtick!

---

## 🎯 **Common Errors & Fixes**

### **Error 1: "Missing expression after unary operator '--'"**

**Problem:** Using backslash `\` instead of backtick `` ` ``

```powershell
# ❌ WRONG (Bash syntax)
python windows_sandbox_client.py \
  --vm-ip 192.168.100.33 \
  --file malware.exe

# ✅ CORRECT (PowerShell syntax)
python windows_sandbox_client.py `
  --vm-ip 192.168.100.33 `
  --file malware.exe

# ✅ OR just use single line
python windows_sandbox_client.py --vm-ip 192.168.100.33 --file malware.exe
```

### **Error 2: "Cannot find path '../Model/...'"**

**Problem:** Using forward slash `/` instead of backslash `\`

```powershell
# ❌ WRONG
python ../Model/predict_lstm_behavioral.py --input features.csv

# ✅ CORRECT
python ..\Model\predict_lstm_behavioral.py --input features.csv
```

### **Error 3: "File not found: malware.exe"**

**Problem:** File is in different directory

```powershell
# Check current directory
pwd

# List files
ls

# Use full path
python windows_sandbox_client.py --vm-ip 192.168.100.33 --file C:\Users\Admin\Desktop\malware.exe

# Or navigate to file location first
cd C:\Users\Admin\Desktop
python windows_sandbox_client.py --vm-ip 192.168.100.33 --file malware.exe
```

---

## 🚀 **Full Example Session**

```powershell
# Navigate to sandbox directory
cd C:\Users\Admin\github-classroom\Caty175\poly_trial\sandbox

# Test connection to VM
python windows_sandbox_client.py --vm-ip 192.168.100.33 --health

# Output:
# ✅ Server is healthy
#    Service: Windows Sandbox API Server
#    OS: win32

# Analyze a malware sample
python windows_sandbox_client.py --vm-ip 192.168.100.33 --file C:\Users\Admin\Desktop\malware.exe --duration 120

# Output:
# ============================================================
# 📤 Sending file to Windows VM for analysis
# ============================================================
# File: malware.exe
# Size: 45,056 bytes
# VM: 192.168.100.33:5000
# Duration: 120 seconds
# ============================================================
#
# 📤 Uploading file...
# ✅ Analysis complete!
#
# ============================================================
# 📊 ANALYSIS RESULTS
# ============================================================
# Analysis ID: abc123-def456-789ghi
# Status: completed
# Timestamp: 2025-11-12T10:30:45
#
# 📈 Summary:
#    API Calls: 156
#    Files Created: 3
#    Files Deleted: 1
#    Libraries Loaded: 12
#    Network Connections: 2
#    DNS Queries: 1
#    Child Processes: 1
# ============================================================
#
# 💾 Report saved: abc123-def456-789ghi_report.json

# Convert to LSTM format
python parse_behavioral_logs.py --input abc123-def456-789ghi_report.json --output features.csv

# Run LSTM prediction
python ..\Model\predict_lstm_behavioral.py --input features.csv

# Output:
# 🤖 LSTM Malware Detection Result
# ============================================================
# Prediction: MALWARE
# Confidence: 94.23%
# ============================================================
```

---

## 📝 **Tips for PowerShell Users**

1. **Use Tab Completion**
   ```powershell
   python windows_sandbox_client.py --vm-ip 192.168.100.33 --file mal<TAB>
   # Autocompletes to: malware.exe
   ```

2. **Use Full Paths to Avoid Confusion**
   ```powershell
   python windows_sandbox_client.py --vm-ip 192.168.100.33 --file C:\Users\Admin\Desktop\malware.exe
   ```

3. **Check Your Current Directory**
   ```powershell
   pwd  # Print working directory
   ls   # List files
   ```

4. **Single Line is Easier**
   ```powershell
   # Instead of multi-line, just use single line:
   python windows_sandbox_client.py --vm-ip 192.168.100.33 --file malware.exe --duration 120
   ```

5. **Save VM IP as Variable**
   ```powershell
   $VM_IP = "192.168.100.33"
   
   # Then use it:
   python windows_sandbox_client.py --vm-ip $VM_IP --file malware.exe
   python windows_sandbox_client.py --vm-ip $VM_IP --list-reports
   ```

---

## 🎯 **Quick Reference Card**

```powershell
# ============================================================
# WINDOWS SANDBOX CLIENT - POWERSHELL COMMANDS
# ============================================================

# Set VM IP (change to your VM's IP)
$VM_IP = "192.168.100.33"

# Test connection
python windows_sandbox_client.py --vm-ip $VM_IP --health

# Analyze file
python windows_sandbox_client.py --vm-ip $VM_IP --file malware.exe --duration 120

# List reports
python windows_sandbox_client.py --vm-ip $VM_IP --list-reports

# Get specific report
python windows_sandbox_client.py --vm-ip $VM_IP --get-report <analysis_id>

# Convert to LSTM format
python parse_behavioral_logs.py --input <id>_report.json --output features.csv

# Run prediction
python ..\Model\predict_lstm_behavioral.py --input features.csv

# ============================================================
```

---

## 🆘 **Need Help?**

```powershell
# Get help for client
python windows_sandbox_client.py --help

# Get help for parser
python parse_behavioral_logs.py --help

# Get help for LSTM predictor
python ..\Model\predict_lstm_behavioral.py --help
```

---

**Remember:** 
- ✅ Use **backtick `` ` ``** for multi-line (or just use single line)
- ✅ Use **backslash `\`** for paths (e.g., `..\Model\`)
- ✅ Replace `192.168.100.33` with your actual VM IP
- ✅ Use full paths if files are in different directories

**Happy malware hunting! 🛡️**

