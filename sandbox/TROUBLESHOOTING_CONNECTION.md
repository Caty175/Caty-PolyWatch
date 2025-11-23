# 🔧 Troubleshooting: Connection Refused Error

**Error:** `ConnectionRefusedError: [WinError 10061] No connection could be made because the target machine actively refused it`

---

## 🎯 **Root Cause**

The error means the **sandbox server is not running** or **not accessible** at the specified IP/port.

---

## ✅ **Quick Fix Steps**

### **Step 1: Check if Server is Running**

```powershell
# In Windows VM, check if server is running:
netstat -an | findstr 5000

# Should show:
# TCP    0.0.0.0:5000    0.0.0.0:0    LISTENING
```

**If nothing shows**, the server is **not running**.

### **Step 2: Start the Sandbox Server**

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

**Keep this terminal open!** The server must stay running.

### **Step 3: Verify Server is Accessible**

```powershell
# From HOST machine, test connection:
python windows_sandbox_client.py --vm-ip 192.168.1.100 --health

# Replace 192.168.1.100 with your VM's actual IP address
# You should see:
# ✅ Server is healthy
```

---

## 🔍 **Common Issues & Solutions**

### **Issue 1: Server Not Started**

**Symptom:** Connection refused error

**Solution:**
```powershell
# Start the server in Windows VM:
python windows_sandbox_server.py --port 5000 --host 0.0.0.0
```

---

### **Issue 2: Wrong IP Address**

**Symptom:** Connection refused, but server is running

**Check VM IP:**
```powershell
# In Windows VM:
ipconfig

# Look for IPv4 Address, e.g.:
# IPv4 Address. . . . . . . . . . . : 192.168.1.100
```

**Update client command:**
```powershell
# Use the correct VM IP:
python windows_sandbox_client.py --vm-ip 192.168.1.100 --file test.exe
```

**If using API server**, set environment variable:
```powershell
# Windows PowerShell:
$env:SANDBOX_VM_IP = "192.168.1.100"

# Or create .env file:
# SANDBOX_VM_IP=192.168.1.100
```

---

### **Issue 3: Firewall Blocking Connection**

**Symptom:** Connection refused even though server is running

**Solution:**
```powershell
# In Windows VM (run as Administrator):
New-NetFirewallRule -DisplayName "Sandbox Server" `
  -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow

# Verify:
Get-NetFirewallRule -DisplayName "Sandbox Server"
```

---

### **Issue 4: Server Running on Wrong Host**

**Symptom:** Connection refused

**Check server is listening on all interfaces:**
```powershell
# Server should be started with:
python windows_sandbox_server.py --host 0.0.0.0 --port 5000

# NOT:
# python windows_sandbox_server.py --host 127.0.0.1 --port 5000  ❌
```

---

### **Issue 5: Using localhost Instead of VM IP**

**Symptom:** Connection to `127.0.0.1:5000` fails

**Problem:** You're trying to connect to localhost, but the server is on the VM.

**Solution:**

**If running client from HOST:**
```powershell
# Use VM IP, not localhost:
python windows_sandbox_client.py --vm-ip 192.168.1.100 --file test.exe
# NOT: --vm-ip 127.0.0.1  ❌
```

**If using API server:**
```powershell
# Set environment variable:
$env:SANDBOX_VM_IP = "192.168.1.100"  # Your VM's IP

# Or in .env file:
# SANDBOX_VM_IP=192.168.1.100
```

---

## 🧪 **Diagnostic Steps**

### **Step 1: Test Network Connectivity**

```powershell
# From HOST, ping VM:
ping 192.168.1.100

# Should get replies
```

### **Step 2: Test Server Health**

```powershell
# From HOST:
python windows_sandbox_client.py --vm-ip 192.168.1.100 --health

# Or using curl:
curl http://192.168.1.100:5000/health
```

### **Step 3: Check Server Logs**

```powershell
# In Windows VM, check server terminal for errors
# Look for:
# - "Server starting..."
# - "Running on http://0.0.0.0:5000"
# - Any error messages
```

### **Step 4: Verify Port is Listening**

```powershell
# In Windows VM:
netstat -an | findstr 5000

# Should show:
# TCP    0.0.0.0:5000    0.0.0.0:0    LISTENING
```

---

## 📋 **Complete Setup Checklist**

Before testing, ensure:

- [ ] **Windows VM is running**
- [ ] **Sandbox server is started** (`windows_sandbox_server.py`)
- [ ] **Server is listening on 0.0.0.0:5000** (not 127.0.0.1)
- [ ] **Firewall allows port 5000** (inbound rule)
- [ ] **VM IP address is known** (use `ipconfig` in VM)
- [ ] **Client uses correct VM IP** (not localhost)
- [ ] **Network connectivity works** (ping VM from host)

---

## 🚀 **Quick Test Sequence**

```powershell
# 1. In Windows VM - Start server:
cd C:\sandbox
python windows_sandbox_server.py --port 5000 --host 0.0.0.0

# 2. In Windows VM - Get IP address:
ipconfig
# Note the IPv4 Address (e.g., 192.168.1.100)

# 3. From HOST - Test connection:
python windows_sandbox_client.py --vm-ip 192.168.1.100 --health

# 4. From HOST - Test file analysis:
python windows_sandbox_client.py --vm-ip 192.168.1.100 --file test.exe
```

---

## 💡 **Pro Tips**

1. **Always start server first** before running client
2. **Use VM IP, not localhost** when connecting from host
3. **Check firewall rules** if connection fails
4. **Verify server is listening** with `netstat`
5. **Keep server terminal open** - closing it stops the server

---

## 🔄 **If Still Not Working**

1. **Check server logs** for error messages
2. **Try different port** (e.g., 5001) to rule out port conflicts
3. **Test with curl** to isolate Python issues
4. **Verify VM network** (bridged vs host-only)
5. **Check VirtualBox network settings**

---

**Most common fix:** Start the sandbox server in the Windows VM! 🚀


