# 🔄 Host ↔ VM Setup Guide

**Send PE files from Host → Windows VM → Receive logs back**

---

## 🎯 **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                        HOST MACHINE                         │
│                     (Your Main Computer)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Run client script:                                      │
│     python windows_sandbox_client.py \                      │
│       --vm-ip 192.168.1.100 \                              │
│       --file malware.exe                                    │
│                                                             │
│  2. Client sends PE file to VM via HTTP                     │
│     └─> POST http://192.168.1.100:5000/analyze             │
│                                                             │
│  3. Client receives analysis report (JSON)                  │
│     └─> Saves: <analysis_id>_report.json                   │
│                                                             │
│  4. Convert to LSTM format:                                 │
│     python parse_behavioral_logs.py \                       │
│       --input report.json --output features.csv             │
│                                                             │
│  5. Run LSTM prediction:                                    │
│     python ../Model/predict_lstm_behavioral.py \            │
│       --input features.csv                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP POST
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       WINDOWS VM                            │
│                  (Isolated Sandbox)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Run server:                                             │
│     python windows_sandbox_server.py --port 5000            │
│                                                             │
│  2. Server receives PE file                                 │
│     └─> Saves to: uploads/<id>_malware.exe                 │
│                                                             │
│  3. Server runs sandbox analysis:                           │
│     └─> python windows_sandbox.py malware.exe              │
│         ├─> Execute malware                                 │
│         ├─> Monitor API calls                               │
│         ├─> Monitor DLLs                                    │
│         ├─> Monitor network                                 │
│         └─> Monitor registry                                │
│                                                             │
│  4. Server generates report:                                │
│     └─> logs/<id>_report.json (320 features)               │
│                                                             │
│  5. Server sends report back to host                        │
│     └─> HTTP Response with JSON                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 **Quick Setup (30 Minutes)**

### **Part 1: Windows VM Setup (20 minutes)**

#### **Step 1: Create Windows VM**

```
1. Download VirtualBox: https://www.virtualbox.org/
2. Download Windows 10 ISO: https://www.microsoft.com/software-download/windows10
3. Create new VM:
   - Name: Windows_Sandbox
   - RAM: 4GB (4096 MB)
   - Disk: 50GB
   - Network: Bridged Adapter (or Host-Only)
4. Install Windows 10
```

#### **Step 2: Configure VM Network**

**Option A: Bridged Network (Easier)**
```
1. In VirtualBox: Settings → Network → Adapter 1
2. Attached to: Bridged Adapter
3. Start VM
4. In Windows VM, open PowerShell:
   ipconfig
5. Note the IPv4 address (e.g., 192.168.1.100)
```

**Option B: Host-Only Network (More Isolated)**
```
1. In VirtualBox: File → Host Network Manager
2. Create new host-only network (e.g., vboxnet0)
3. Note the IP range (e.g., 192.168.56.0/24)
4. In VirtualBox: Settings → Network → Adapter 1
5. Attached to: Host-only Adapter
6. Name: vboxnet0
7. Start VM
8. In Windows VM:
   ipconfig
9. Note the IPv4 address (e.g., 192.168.56.101)
```

#### **Step 3: Install Python in VM**

```powershell
# In Windows VM:

# 1. Download Python 3.11+
# Visit: https://www.python.org/downloads/

# 2. Install Python (check "Add to PATH")

# 3. Verify installation
python --version

# 4. Install dependencies
pip install psutil pywin32 wmi flask requests
```

#### **Step 4: Copy Files to VM**

```powershell
# Option A: Shared Folder (VirtualBox)
# 1. In VirtualBox: Settings → Shared Folders
# 2. Add folder: C:\Users\Admin\github-classroom\Caty175\poly_trial\sandbox
# 3. In Windows VM, access: \\VBOXSVR\sandbox

# Option B: Manual Copy
# 1. Copy these files to VM:
#    - windows_sandbox.py
#    - windows_sandbox_server.py
# 2. Save to: C:\sandbox\
```

#### **Step 5: Configure Windows Firewall**

```powershell
# In Windows VM, run as Administrator:

# Allow port 5000
New-NetFirewallRule -DisplayName "Sandbox Server" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow

# Verify
Get-NetFirewallRule -DisplayName "Sandbox Server"
```

#### **Step 6: Start Sandbox Server**

```powershell
# In Windows VM:
cd C:\sandbox
python windows_sandbox_server.py --port 5000 --host 0.0.0.0

# You should see:
# ============================================================
# 🪟 Windows Sandbox API Server
# ============================================================
# Host: 0.0.0.0
# Port: 5000
# ============================================================
# ✅ Server starting...
```

---

### **Part 2: Host Machine Setup (10 minutes)**

#### **Step 1: Install Client Dependencies**

```bash
# On your host machine (can be Windows, Linux, or Mac):
pip install requests
```

#### **Step 2: Test Connection**

```bash
# Replace 192.168.1.100 with your VM's IP
python windows_sandbox_client.py --vm-ip 192.168.1.100 --health

# You should see:
# ✅ Server is healthy
#    Service: Windows Sandbox API Server
#    OS: win32
```

#### **Step 3: Analyze a File**

**PowerShell (Windows Host):**
```powershell
# Send malware to VM for analysis (single line)
python windows_sandbox_client.py --vm-ip 192.168.1.100 --file malware.exe --duration 120

# Or multi-line (use backtick ` for line continuation):
python windows_sandbox_client.py `
  --vm-ip 192.168.1.100 `
  --file malware.exe `
  --duration 120
```

**Bash (Linux/Mac Host):**
```bash
# Send malware to VM for analysis (use backslash \ for line continuation)
python windows_sandbox_client.py \
  --vm-ip 192.168.1.100 \
  --file malware.exe \
  --duration 120
```

**What happens:**
```
# The client will:
# 1. Upload malware.exe to VM
# 2. Wait for analysis (120 seconds)
# 3. Download report.json
# 4. Save locally: <analysis_id>_report.json
```

---

## 📋 **Complete Workflow**

### **Daily Usage**

**PowerShell (Windows Host):**
```powershell
# 1. Start Windows VM
# (In VirtualBox, start the VM)

# 2. In VM: Start sandbox server
python windows_sandbox_server.py --port 5000

# 3. On Host: Analyze malware
python windows_sandbox_client.py --vm-ip 192.168.1.100 --file malware.exe --duration 120

# 4. On Host: Convert to LSTM format
python parse_behavioral_logs.py --input <analysis_id>_report.json --output features.csv

# 5. On Host: Run LSTM prediction
python ..\Model\predict_lstm_behavioral.py --input features.csv

# Output:
# 🤖 LSTM Malware Detection Result
# ============================================================
# Prediction: MALWARE
# Confidence: 94.23%
# ============================================================

# 6. In VM: Revert to snapshot (optional)
# (In VirtualBox: Snapshots → Restore)
```

**Bash (Linux/Mac Host):**
```bash
# 1. Start Windows VM
# (In VirtualBox, start the VM)

# 2. In VM: Start sandbox server
python windows_sandbox_server.py --port 5000

# 3. On Host: Analyze malware
python windows_sandbox_client.py \
  --vm-ip 192.168.1.100 \
  --file malware.exe \
  --duration 120

# 4. On Host: Convert to LSTM format
python parse_behavioral_logs.py \
  --input <analysis_id>_report.json \
  --output features.csv

# 5. On Host: Run LSTM prediction
python ../Model/predict_lstm_behavioral.py \
  --input features.csv

# Output:
# 🤖 LSTM Malware Detection Result
# ============================================================
# Prediction: MALWARE
# Confidence: 94.23%
# ============================================================

# 6. In VM: Revert to snapshot (optional)
# (In VirtualBox: Snapshots → Restore)
```

---

## 🔧 **Troubleshooting**

### **Issue 1: Cannot connect to VM**

```bash
# Test 1: Ping VM
ping 192.168.1.100

# Test 2: Check if server is running
curl http://192.168.1.100:5000/health

# Test 3: Check firewall
# In VM PowerShell:
Get-NetFirewallRule -DisplayName "Sandbox Server"

# Test 4: Check server is listening
# In VM PowerShell:
netstat -an | findstr 5000
```

### **Issue 2: File upload fails**

```bash
# Check file exists
ls -l malware.exe

# Check file size (max ~100MB recommended)
# Large files may timeout

# Increase timeout in client
# Edit windows_sandbox_client.py:
# timeout=duration + 120  # Increase buffer
```

### **Issue 3: Analysis timeout**

```bash
# Increase duration
python windows_sandbox_client.py \
  --vm-ip 192.168.1.100 \
  --file malware.exe \
  --duration 300  # 5 minutes

# Or retrieve report later
python windows_sandbox_client.py \
  --vm-ip 192.168.1.100 \
  --list-reports

python windows_sandbox_client.py \
  --vm-ip 192.168.1.100 \
  --get-report <analysis_id>
```

### **Issue 4: VM IP keeps changing**

```powershell
# In Windows VM, set static IP:

# 1. Open Network Settings
# 2. Change adapter settings
# 3. Right-click adapter → Properties
# 4. IPv4 → Properties
# 5. Use the following IP address:
#    IP: 192.168.1.100
#    Subnet: 255.255.255.0
#    Gateway: 192.168.1.1
```

---

## 📊 **API Reference**

### **Server Endpoints (VM)**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze file (upload) |
| `/report/<id>` | GET | Get report by ID |
| `/reports` | GET | List all reports |
| `/config` | GET/POST | Get/update config |

### **Client Commands (Host)**

```bash
# Health check
python windows_sandbox_client.py --vm-ip <IP> --health

# Analyze file
python windows_sandbox_client.py --vm-ip <IP> --file <file>

# List reports
python windows_sandbox_client.py --vm-ip <IP> --list-reports

# Get specific report
python windows_sandbox_client.py --vm-ip <IP> --get-report <id>
```

---

## 🔒 **Security Notes**

1. **VM Isolation**: Always run in isolated VM
2. **Network**: Use host-only network for maximum isolation
3. **Snapshots**: Take snapshots before analysis
4. **Firewall**: Only allow port 5000 from host
5. **No Internet**: Disable internet in VM (optional but recommended)

---

## ✅ **Summary**

You now have a complete **Host ↔ VM** setup:

- ✅ **Windows VM** runs sandbox server
- ✅ **Host machine** sends files via client
- ✅ **Automatic analysis** with 320 features
- ✅ **Reports sent back** to host
- ✅ **Ready for LSTM** prediction

**Total setup time**: ~30 minutes  
**Per-sample analysis**: ~2-5 minutes  

---

**Quick Start Commands:**

```bash
# In VM:
python windows_sandbox_server.py --port 5000

# On Host:
python windows_sandbox_client.py --vm-ip 192.168.1.100 --file malware.exe
```

🎉 **You're ready to analyze malware safely!**

