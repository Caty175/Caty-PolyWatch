# 🛡️ Malware Analysis Sandbox for LSTM Model

**Practical, manual sandbox for Ubuntu/Linux that captures 320 behavioral features for LSTM malware detection**

---

## 📋 **What This Does**

This is a **simple, practical sandbox** that runs on Ubuntu/Linux and captures behavioral features for your LSTM model:

```
[API/Frontend] → [Ubuntu Sandbox VM]
                 │
                 ├─ Receives uploaded file
                 ├─ Runs in isolated sandbox (Firejail/chroot/container)
                 ├─ Logs system calls (strace)
                 ├─ Captures network activity (tcpdump)
                 ├─ Extracts 320 features
                 └─ Returns JSON report
```

**Key Features:**
- ✅ **No Windows VM required** - Runs on Ubuntu/Linux
- ✅ **Manual control** - Simple scripts you can understand and modify
- ✅ **REST API** - Easy integration with frontend
- ✅ **320 features** - Complete LSTM model compatibility
- ✅ **Multiple isolation options** - Firejail, Docker, or basic

---

## 🎯 **Feature Coverage**

Your LSTM model requires **320 behavioral features**:

| Category | Count | Description |
|----------|-------|-------------|
| **API Calls** | 261 | Mapped from Linux syscalls to Windows APIs |
| **File Operations** | 5 | File created/deleted/read/written/opened |
| **Library Loading** | 50 | Frequency of libraries loaded (.so → .dll) |
| **Behavioral** | 4 | Registry, network, directory operations |
| **TOTAL** | **320** | Complete behavioral profile |

---

## ⚡ **Quick Start (5 Minutes)**

### **Step 1: Install Dependencies**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required tools
sudo apt install -y python3 python3-pip strace tcpdump firejail

# Install Python packages
pip3 install flask pandas numpy
```

### **Step 2: Test the Sandbox**

```bash
cd ~/poly_trial/sandbox

# Run test script
chmod +x test_sandbox.sh
./test_sandbox.sh
```

### **Step 3: Start API Server**

```bash
# Start REST API server
python3 sandbox_api.py --port 5000
```

### **Step 4: Submit a File**

```bash
# In another terminal
curl -X POST -F "file=@test.sh" http://localhost:5000/analyze
```

**That's it!** Your sandbox is now running and ready to analyze files.

---

## 📦 **What's Included**

### **Core Components**

| File | Purpose |
|------|---------|
| **`sandbox.py`** | Core sandbox engine - runs files and captures behavior |
| **`sandbox_api.py`** | REST API server - receives files via HTTP |
| **`parse_behavioral_logs.py`** | Feature parser - converts to LSTM format |
| **`test_sandbox.sh`** | Test script - verifies setup |

### **Documentation**

| File | Purpose |
|------|---------|
| **`UBUNTU_SANDBOX_GUIDE.md`** | Complete setup and usage guide |
| **`FEATURE_REQUIREMENTS.md`** | 320 feature specification |
| **`FEATURES_SUMMARY.txt`** | Quick reference |
| **`QUICK_START.md`** | 15-minute setup (Windows VM approach) |
| **`COMPLETE_GUIDE.md`** | Comprehensive guide (all options) |

---

## 🚀 **Usage Options**

### **Option 1: REST API (Recommended)**

**Start server:**
```bash
python3 sandbox_api.py --port 5000
```

**Submit file:**
```bash
curl -X POST -F "file=@malware.bin" http://localhost:5000/analyze
```

**Get results:**
```bash
curl http://localhost:5000/result/<analysis_id>
```

**Perfect for:** Web frontends, automated pipelines, remote analysis

---

### **Option 2: Command Line**

**Basic usage:**
```bash
python3 sandbox.py malware.bin
```

**With options:**
```bash
python3 sandbox.py malware.bin --duration 180 --output report.json
```

**Perfect for:** Manual analysis, testing, debugging

---

### **Option 3: Python Integration**

```python
from sandbox import LinuxSandbox

# Create sandbox
sandbox = LinuxSandbox(duration=120, use_firejail=True)

# Run analysis
report = sandbox.run_in_sandbox('malware.bin', 'report.json')

# Access features
print(f"API calls: {len(report['api_calls'])}")
print(f"Files created: {report['file_operations']['file_created']}")
```

**Perfect for:** Custom integrations, batch processing

---

## 📊 **Complete Workflow**

### **1. Run Sandbox Analysis**

```bash
python3 sandbox.py malware.bin --output report.json
```

**Output:** `report.json` with behavioral data

### **2. Convert to LSTM Format**

```bash
python3 parse_behavioral_logs.py \
    --input report.json \
    --output features.csv \
    --metadata ../Model/components/lstm_model_metadata.json
```

**Output:** `features.csv` with 320 features

### **3. Run LSTM Prediction**

```bash
python3 ../Model/predict_lstm_behavioral.py --input features.csv
```

**Output:**
```
🤖 LSTM Malware Detection Result
============================================================
Prediction: MALWARE
Confidence: 94.23%
Malware Probability: 94.23%
```

---

## 🔧 **How It Works**

### **Syscall Mapping**

The sandbox maps Linux syscalls to Windows API equivalents:

| Linux Syscall | Windows API | Category |
|---------------|-------------|----------|
| `open`, `openat` | `API_NtOpenFile` | File I/O |
| `creat` | `API_NtCreateFile` | File I/O |
| `read` | `API_NtReadFile` | File I/O |
| `write` | `API_NtWriteFile` | File I/O |
| `fork`, `execve` | `API_CreateProcessInternalW` | Process |
| `mmap` | `API_NtAllocateVirtualMemory` | Memory |
| `socket`, `connect` | `API_socket`, `API_connect` | Network |
| `dlopen` | `API_LdrLoadDll` | Library |

**See:** `sandbox.py` for complete mapping (100+ syscalls)

### **Feature Extraction**

```python
# 1. Capture syscalls with strace
strace -o syscalls.log -f -e trace=all ./malware.bin

# 2. Parse syscalls
syscalls = parse_syscall_log('syscalls.log')

# 3. Map to Windows APIs
api_calls = map_syscalls_to_apis(syscalls)
# Result: {'API_NtOpenFile': 45, 'API_NtWriteFile': 23, ...}

# 4. Extract file operations
file_ops = extract_file_operations(syscalls)
# Result: {'file_created': 12, 'file_deleted': 3, ...}

# 5. Extract library loading
dll_loaded = extract_library_loading(syscalls)
# Result: {'kernel32.dll': 156, 'ntdll.dll': 234, ...}

# 6. Extract behavioral indicators
behavioral = extract_behavioral_indicators(syscalls)
# Result: {'regkey_read': 34, 'resolves_host': 5, ...}
```

---

## 🔒 **Isolation Options**

### **Firejail (Recommended)**

```bash
python3 sandbox.py malware.bin --firejail
```

**Features:**
- ✅ Filesystem isolation
- ✅ Network isolation
- ✅ Resource limits
- ✅ Easy to use

### **Docker**

```bash
python3 sandbox.py malware.bin --docker
```

**Features:**
- ✅ Complete isolation
- ✅ Reproducible environment
- ❌ Requires Docker daemon

### **No Isolation (Testing Only)**

```bash
python3 sandbox.py test.sh --no-isolation
```

**⚠️ WARNING:** Only use for benign files!

---

## 📡 **REST API Reference**

### **Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API documentation |
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Submit file for analysis |
| `GET` | `/status/<id>` | Get analysis status |
| `GET` | `/result/<id>` | Get analysis result |
| `GET` | `/download/<id>` | Download full report |

### **Example: Submit File**

**Request:**
```bash
curl -X POST \
  -F "file=@malware.bin" \
  -F "duration=120" \
  http://localhost:5000/analyze
```

**Response:**
```json
{
  "analysis_id": "abc123-def456-...",
  "status": "queued",
  "message": "Analysis started",
  "estimated_completion": "120 seconds"
}
```

### **Example: Get Results**

**Request:**
```bash
curl http://localhost:5000/result/abc123-def456-...
```

**Response:**
```json
{
  "analysis_id": "abc123-def456-...",
  "status": "completed",
  "summary": {
    "api_calls": 87,
    "files_created": 12,
    "files_deleted": 3,
    "libraries_loaded": 23,
    "dns_queries": 5
  },
  "report": {
    "metadata": {...},
    "api_calls": {...},
    "file_operations": {...},
    "dll_loaded": {...},
    "behavioral_indicators": {...}
  }
}
```

---

## 🐛 **Troubleshooting**

### **Issue: "strace: command not found"**

```bash
sudo apt install strace
```

### **Issue: "tcpdump: permission denied"**

```bash
sudo usermod -aG pcap $USER
# Or run with sudo
sudo python3 sandbox.py malware.bin
```

### **Issue: "firejail: command not found"**

```bash
sudo apt install firejail
# Or use --no-isolation for testing
python3 sandbox.py test.sh --no-isolation
```

### **Issue: "Few features captured"**

This is normal for simple scripts. For better results:
- Increase duration: `--duration 300`
- Use actual malware samples
- Check if file is executable: `chmod +x file`

### **Issue: "Flask not found"**

```bash
pip3 install flask pandas numpy
```

---

## 📚 **Documentation**

| Document | Purpose |
|----------|---------|
| **[UBUNTU_SANDBOX_GUIDE.md](UBUNTU_SANDBOX_GUIDE.md)** | Complete Ubuntu/Linux setup guide |
| **[FEATURE_REQUIREMENTS.md](FEATURE_REQUIREMENTS.md)** | 320 feature specification |
| **[FEATURES_SUMMARY.txt](FEATURES_SUMMARY.txt)** | Quick reference |
| **[QUICK_START.md](QUICK_START.md)** | Windows VM approach (alternative) |
| **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** | All options and approaches |

---

## ⚠️ **Security Warnings**

1. **ALWAYS** run unknown files in isolated environment
2. **NEVER** disable isolation for untrusted files
3. **ALWAYS** use network isolation
4. **NEVER** run on production systems
5. **ALWAYS** review logs before sharing

---

## 🎯 **Next Steps**

1. ✅ **Test setup:** `./test_sandbox.sh`
2. ✅ **Start API:** `python3 sandbox_api.py --port 5000`
3. ✅ **Submit test file:** `curl -X POST -F "file=@test.sh" http://localhost:5000/analyze`
4. ✅ **Review docs:** [UBUNTU_SANDBOX_GUIDE.md](UBUNTU_SANDBOX_GUIDE.md)
5. ✅ **Integrate with frontend:** Use REST API endpoints

---

## 📞 **Support**

**For issues:**
1. Run test script: `./test_sandbox.sh`
2. Check [UBUNTU_SANDBOX_GUIDE.md](UBUNTU_SANDBOX_GUIDE.md)
3. Review logs in `logs/` directory
4. Test with `--no-isolation` for debugging

**For questions:**
- Check documentation in `sandbox/` directory
- Review `sandbox.py` comments
- See examples in guides

---

**Ready to analyze! 🛡️**

Start with: `python3 sandbox_api.py --port 5000`

### **Option 2: Manual Setup (Full Control)**

See [SANDBOX_SETUP_GUIDE.md](SANDBOX_SETUP_GUIDE.md) for detailed instructions.

**Quick version:**

```bash
# 1. Set up Windows VM with monitoring script
# Copy windows_monitor.py to Windows VM

# 2. On Windows VM - Run analysis
python windows_monitor.py --target malware.exe --duration 120 --output report.json

# 3. On REMnux - Parse features
python parse_behavioral_logs.py --input report.json --output features.csv --summary

# 4. Run LSTM prediction
python ../Model/predict_lstm_behavioral.py --input features.csv
```

---

## 📁 **Files Overview**

```
sandbox/
├── README.md                      # This file
├── SANDBOX_SETUP_GUIDE.md         # Detailed setup instructions
├── windows_monitor.py             # Behavioral monitor (runs on Windows VM)
├── parse_behavioral_logs.py       # Convert logs to LSTM format
├── docker-compose.yml             # Docker deployment
├── analyze_sample.sh              # Automated analysis script
└── samples/                       # Place malware samples here
```

---

## 🔧 **Components**

### **1. Windows Monitor (`windows_monitor.py`)**

Runs on Windows analysis VM to capture behavioral data.

**Features:**
- ✅ Process monitoring (PID, children, command line)
- ✅ File operations tracking
- ✅ DLL loading detection
- ✅ Network connection monitoring
- ✅ Registry access tracking (Windows only)

**Usage:**
```bash
python windows_monitor.py --target malware.exe --duration 120 --output report.json
```

**Output:** JSON report with behavioral data

### **2. Log Parser (`parse_behavioral_logs.py`)**

Converts behavioral reports to LSTM-compatible CSV format.

**Features:**
- ✅ Extracts 320 features from JSON report
- ✅ Maps to LSTM model feature names
- ✅ Handles missing features gracefully
- ✅ Generates human-readable summary

**Usage:**
```bash
python parse_behavioral_logs.py \
    --input report.json \
    --output features.csv \
    --metadata ../Model/components/lstm_model_metadata.json \
    --summary
```

**Output:** CSV with 320 features ready for LSTM model

### **3. Automated Analysis (`analyze_sample.sh`)**

End-to-end automation script for VirtualBox-based analysis.

**Features:**
- ✅ Restores VM to clean snapshot
- ✅ Transfers sample to VM
- ✅ Runs behavioral monitor
- ✅ Retrieves report
- ✅ Parses features
- ✅ Runs LSTM prediction

**Usage:**
```bash
./analyze_sample.sh malware.exe 120
```

---

## 📊 **Workflow**

```
┌─────────────────┐
│  Malware Sample │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Windows Analysis VM    │
│  (windows_monitor.py)   │
│  - Execute sample       │
│  - Monitor APIs         │
│  - Track file ops       │
│  - Log DLL loading      │
│  - Capture network      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Behavioral Report      │
│  (JSON format)          │
│  - API calls            │
│  - File operations      │
│  - DLL loaded           │
│  - Network activity     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature Parser         │
│  (parse_behavioral_     │
│   logs.py)              │
│  - Extract 320 features │
│  - Format for LSTM      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature CSV            │
│  (320 columns)          │
│  - 261 API features     │
│  - 5 file op features   │
│  - 50 DLL features      │
│  - 4 behavioral         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  LSTM Model             │
│  (predict_lstm_         │
│   behavioral.py)        │
│  - Load model           │
│  - Normalize features   │
│  - Predict malware      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Prediction Result      │
│  - Benign/Malware       │
│  - Confidence score     │
│  - Probability          │
└─────────────────────────┘
```

---

## 🔍 **Example Output**

### **Behavioral Report Summary**

```
📊 BEHAVIORAL ANALYSIS SUMMARY
============================================================

File Operations:
  • Created: 12
  • Deleted: 3
  • Read: 45
  • Written: 28
  • Opened: 67

DLL Loading:
  • Total DLLs: 23
  • kernel32.dll: 156
  • ntdll.dll: 234
  • ws2_32.dll: 12

Behavioral Indicators:
  • Registry Keys Read: 34
  • Directories Enumerated: 8
  • Network Connections: 5

API Calls:
  • Total Unique APIs: 87
  • Total API Calls: 1,234
```

### **LSTM Prediction**

```
🤖 LSTM Malware Detection Result
============================================================
Prediction: MALWARE
Confidence: 94.23%
Malware Probability: 94.23%
Benign Probability: 5.77%

⚠️  WARNING: This sample exhibits malicious behavior!
```

---

## 🛠️ **Installation**

### **Prerequisites**

- **REMnux** or Ubuntu 20.04+
- **VirtualBox** 6.0+
- **Python 3.8+**
- **Windows 7/10 VM** for analysis

### **Install Dependencies**

```bash
# On REMnux/Ubuntu
sudo apt update
sudo apt install -y python3 python3-pip virtualbox

# Install Python packages
pip3 install pandas numpy psutil

# On Windows VM
pip install pywin32 psutil
```

---

## 📚 **Documentation**

- **[SANDBOX_SETUP_GUIDE.md](SANDBOX_SETUP_GUIDE.md)** - Complete setup instructions
- **[../Model/LSTM_BEHAVIORAL_README.md](../Model/LSTM_BEHAVIORAL_README.md)** - LSTM model documentation
- **[../QUICK_START.md](../QUICK_START.md)** - Project quick start guide

---

## ⚠️ **Security Warning**

**DANGER:** This system executes real malware!

- ✅ Always use isolated VMs
- ✅ Use host-only networking (no internet access)
- ✅ Take snapshots before analysis
- ✅ Never run on production systems
- ✅ Keep analysis VMs offline

---

## 🐛 **Troubleshooting**

### **Issue: Monitor doesn't capture API calls**

The basic monitor uses `psutil` which has limited API visibility. For full API monitoring:

1. **Use CAPE Sandbox** (recommended)
2. **Use API Monitor tool** on Windows
3. **Use Frida** for dynamic instrumentation

### **Issue: Missing features in CSV**

```bash
# Check if metadata file exists
ls -la ../Model/components/lstm_model_metadata.json

# Run parser with --summary to see what's captured
python parse_behavioral_logs.py --input report.json --summary
```

### **Issue: LSTM prediction fails**

```bash
# Verify feature count
python -c "import pandas as pd; df = pd.read_csv('features.csv'); print(df.shape)"

# Should output: (1, 320) or similar
```

---

## 🎯 **Next Steps**

1. ✅ Read [SANDBOX_SETUP_GUIDE.md](SANDBOX_SETUP_GUIDE.md)
2. ✅ Set up Windows analysis VM
3. ✅ Test with known malware samples
4. ✅ Verify 320 features are captured
5. ✅ Integrate with your LSTM model
6. ✅ Automate analysis pipeline

---

## 📞 **Support**

For issues or questions:
1. Check [SANDBOX_SETUP_GUIDE.md](SANDBOX_SETUP_GUIDE.md)
2. Review LSTM model docs in `Model/`
3. Check script comments for usage examples

---

**Ready to analyze malware? Start with the [SANDBOX_SETUP_GUIDE.md](SANDBOX_SETUP_GUIDE.md)!**

