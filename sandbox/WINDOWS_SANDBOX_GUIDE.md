# 🪟 Windows Sandbox Setup Guide

**Native Windows sandbox for accurate LSTM malware analysis**

---

## 🎯 **Why Windows Sandbox?**

Your LSTM model was **trained on Windows malware behavior**, so a Windows-based sandbox provides:

✅ **Real Windows API calls** - No approximation needed  
✅ **Accurate DLL tracking** - Real Windows libraries  
✅ **Native PE execution** - Run actual Windows malware  
✅ **Registry monitoring** - Windows-specific features  
✅ **Better accuracy** - Matches training data exactly  

---

## 📋 **Requirements**

### **System Requirements**

- **OS**: Windows 10/11 or Windows Server 2016+
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 20GB free space
- **Network**: Isolated network (host-only or no internet)

### **Software Requirements**

```powershell
# Python 3.7+
python --version

# Required packages
pip install psutil pywin32 wmi
```

---

## ⚡ **Quick Start (5 Minutes)**

### **Step 1: Install Dependencies**

```powershell
# Install Python packages
pip install psutil pywin32 wmi

# Verify installation
python -c "import psutil, win32api, wmi; print('✅ All modules installed')"
```

### **Step 2: Test the Sandbox**

```powershell
# Navigate to sandbox directory
cd C:\Users\Admin\github-classroom\Caty175\poly_trial\sandbox

# Test with a benign file (e.g., notepad)
python windows_sandbox.py C:\Windows\System32\notepad.exe --duration 10
```

### **Step 3: Analyze Malware**

```powershell
# Analyze malware sample
python windows_sandbox.py malware.exe --duration 120 --output report.json
```

**That's it!** Your Windows sandbox is now capturing real API calls.

---

## 🚀 **Usage**

### **Basic Usage**

```powershell
python windows_sandbox.py malware.exe
```

### **With Options**

```powershell
# Custom duration (3 minutes)
python windows_sandbox.py malware.exe --duration 180

# Custom output file
python windows_sandbox.py malware.exe --output my_report.json

# Custom log directory
python windows_sandbox.py malware.exe --log-dir C:\analysis\logs
```

### **Complete Pipeline**

```powershell
# 1. Run sandbox analysis
python windows_sandbox.py malware.exe --output report.json

# 2. Convert to LSTM format
python parse_behavioral_logs.py --input report.json --output features.csv

# 3. Run LSTM prediction
python ..\Model\predict_lstm_behavioral.py --input features.csv
```

---

## 📊 **What It Captures**

### **1. Windows API Calls (261 features)**

Real Windows APIs captured:
- `API_NtCreateFile` - File creation
- `API_NtOpenFile` - File opening
- `API_NtReadFile` - File reading
- `API_NtWriteFile` - File writing
- `API_CreateProcessInternalW` - Process creation
- `API_LdrLoadDll` - DLL loading
- `API_RegOpenKeyExA` - Registry access
- `API_socket` - Network socket creation
- `API_connect` - Network connections
- ... and 252 more APIs

### **2. File Operations (5 features)**

- `file_created` - Files created by malware
- `file_deleted` - Files deleted
- `file_read` - Files read
- `file_written` - Files written
- `file_opened` - Files opened

### **3. DLL Loading (50 features)**

Real Windows DLLs tracked:
- `kernel32.dll` - Core Windows functions
- `ntdll.dll` - NT kernel layer
- `advapi32.dll` - Advanced APIs (registry, security)
- `user32.dll` - GUI functions
- `ws2_32.dll` - Windows Sockets
- ... and 45 more DLLs

### **4. Behavioral Indicators (4 features)**

- `regkey_read` - Registry key reads
- `directory_enumerated` - Directory listings
- `dll_loaded_count` - Total DLLs loaded
- `resolves_host` - DNS queries

---

## 🔧 **How It Works**

### **Architecture**

```
┌─────────────────────────────────────────────────┐
│           Windows Sandbox                       │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Execute malware.exe                         │
│     └─> CreateProcess() → Get PID              │
│                                                 │
│  2. Start Monitoring Threads                    │
│     ├─> File Monitor (psutil)                   │
│     ├─> DLL Monitor (memory_maps)               │
│     ├─> Network Monitor (connections)           │
│     ├─> Registry Monitor (WMI)                  │
│     └─> Process Monitor (children)              │
│                                                 │
│  3. Capture Behavior (120 seconds)              │
│     └─> Track API calls, file ops, DLLs, etc.  │
│                                                 │
│  4. Terminate Process                           │
│     └─> Kill malware and children               │
│                                                 │
│  5. Generate JSON Report                        │
│     └─> 320 features ready for LSTM             │
│                                                 │
└─────────────────────────────────────────────────┘
```

### **Monitoring Methods**

| Feature | Method | Library |
|---------|--------|---------|
| **Process Creation** | `CreateProcess()` | `win32process` |
| **File Operations** | `open_files()`, `io_counters()` | `psutil` |
| **DLL Loading** | `memory_maps()` | `psutil` |
| **Network Activity** | `connections()` | `psutil` |
| **Registry Access** | WMI events | `wmi` |
| **Child Processes** | `children()` | `psutil` |

---

## 📁 **Output Format**

### **JSON Report Structure**

```json
{
  "metadata": {
    "analysis_id": "abc123-def456-...",
    "target": "malware.exe",
    "timestamp": "2025-11-11T10:00:00",
    "duration": 120,
    "os": "Windows 10 10.0.19045",
    "pid": 1234,
    "child_pids": [5678, 9012]
  },
  "api_calls": {
    "API_NtCreateFile": 45,
    "API_NtOpenFile": 123,
    "API_LdrLoadDll": 67,
    "API_CreateProcessInternalW": 2,
    ...
  },
  "file_operations": {
    "file_created": 12,
    "file_deleted": 3,
    "file_read": 45,
    "file_written": 28,
    "file_opened": 67
  },
  "dll_loaded": {
    "kernel32.dll": 156,
    "ntdll.dll": 234,
    "advapi32.dll": 89,
    ...
  },
  "behavioral_indicators": {
    "regkey_read": 34,
    "directory_enumerated": 8,
    "dll_loaded_count": 23,
    "resolves_host": 5,
    "command_line": 1
  },
  "network_activity": [
    {
      "type": "connection",
      "protocol": "TCP",
      "local": "192.168.1.100:49152",
      "remote": "8.8.8.8:53",
      "status": "ESTABLISHED"
    }
  ],
  "summary": {
    "api_calls": 87,
    "files_created": 12,
    "libraries_loaded": 23,
    "network_connections": 5,
    "dns_queries": 2,
    "child_processes": 2
  }
}
```

---

## 🔒 **Security Best Practices**

### **⚠️ CRITICAL: Isolation is Essential!**

1. **Use a Virtual Machine**
   - VMware Workstation
   - VirtualBox
   - Hyper-V
   - Windows Sandbox (built-in)

2. **Network Isolation**
   ```powershell
   # Disable network adapter
   Disable-NetAdapter -Name "Ethernet" -Confirm:$false
   
   # Or use host-only network
   # Configure in VM settings
   ```

3. **Snapshot Before Analysis**
   - Take VM snapshot
   - Run analysis
   - Revert to snapshot after

4. **Never Run on Host Machine**
   - ❌ Don't run on your main PC
   - ❌ Don't run on production servers
   - ✅ Always use isolated VM

---

## 🐛 **Troubleshooting**

### **Issue: "Module not found: win32api"**

```powershell
pip install pywin32

# If still fails, run post-install script
python Scripts\pywin32_postinstall.py -install
```

### **Issue: "Access Denied" errors**

```powershell
# Run as Administrator
# Right-click PowerShell → "Run as Administrator"
python windows_sandbox.py malware.exe
```

### **Issue: "Process terminated immediately"**

- Malware may have anti-sandbox checks
- Try increasing duration: `--duration 300`
- Check if file is actually executable
- Review logs for errors

### **Issue: "Few API calls captured"**

This is normal for simple programs. For better results:
- Use actual malware samples
- Increase duration
- Check if process is actually running

### **Issue: "WMI errors"**

```powershell
# Restart WMI service
net stop winmgmt
net start winmgmt
```

---

## 📊 **Comparison: Windows vs Linux Sandbox**

| Feature | Windows Sandbox | Linux Sandbox |
|---------|----------------|---------------|
| **API Accuracy** | ✅ Real Windows APIs | ⚠️ Mapped from syscalls |
| **DLL Tracking** | ✅ Real Windows DLLs | ⚠️ Mapped from .so files |
| **PE Execution** | ✅ Native execution | ❌ Can't run PE files |
| **Registry** | ✅ Real registry access | ❌ No registry |
| **Model Accuracy** | ✅ Matches training data | ⚠️ Approximate |
| **Setup Complexity** | ⚠️ Requires Windows VM | ✅ Simple (Ubuntu) |
| **Use Case** | Windows malware | Scripts, cross-platform |

**Recommendation**: Use **Windows sandbox** for real Windows malware analysis!

---

## 📚 **Next Steps**

1. ✅ **Set up Windows VM** (if not already done)
2. ✅ **Install dependencies**: `pip install psutil pywin32 wmi`
3. ✅ **Test with benign file**: `python windows_sandbox.py notepad.exe --duration 10`
4. ✅ **Analyze malware**: `python windows_sandbox.py malware.exe --duration 120`
5. ✅ **Convert to LSTM format**: `python parse_behavioral_logs.py --input report.json --output features.csv`
6. ✅ **Run prediction**: `python ..\Model\predict_lstm_behavioral.py --input features.csv`

---

## 🎯 **Example Workflow**

```powershell
# 1. Take VM snapshot
# (Do this in your VM software)

# 2. Run sandbox analysis
python windows_sandbox.py C:\samples\malware.exe --output report.json

# 3. Convert to LSTM format
python parse_behavioral_logs.py --input report.json --output features.csv

# 4. Run LSTM prediction
python ..\Model\predict_lstm_behavioral.py --input features.csv

# Output:
# 🤖 LSTM Malware Detection Result
# ============================================================
# Prediction: MALWARE
# Confidence: 94.23%
# ============================================================

# 5. Revert VM to snapshot
# (Do this in your VM software)
```

---

**Ready to analyze Windows malware! 🪟🛡️**

**Start with**: `python windows_sandbox.py malware.exe --duration 120`

