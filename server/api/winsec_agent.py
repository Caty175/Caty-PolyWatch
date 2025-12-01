# winsec_agent.py
import os
import time
import platform
import subprocess
import json
import requests
import hashlib
import pefile
import psutil
from pathlib import Path
from datetime import datetime, timezone

SERVER_URL = os.environ.get("WINSEC_SERVER", "http://127.0.0.1:8000")
API_KEY = os.environ.get("WINSEC_API_KEY", "replace_with_secure_key")
HOST_ID = os.environ.get("WINSEC_HOST_ID", platform.node().lower() + "_" + str(os.getpid()))

HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def get_defender_status():
    # Uses PowerShell Get-MpComputerStatus if available
    try:
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
               "try { (Get-MpComputerStatus | Select-Object AMServiceEnabled,RealTimeProtectionEnabled,AntispywareEnabled) | ConvertTo-Json -Depth 1 } catch { Write-Output '{}'}"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=8, text=True)
        j = json.loads(out) if out.strip() else {}
        # If object is empty, return None/False
        enabled = j.get("AMServiceEnabled") or j.get("RealTimeProtectionEnabled") or j.get("AntispywareEnabled")
        return bool(enabled)
    except Exception:
        return None

def get_firewall_status():
    try:
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
               "try { Get-NetFirewallProfile | Select-Object Name,Enabled | ConvertTo-Json -Depth 1 } catch { Write-Output '[]' }"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=6, text=True)
        j = json.loads(out)
        # If any profile shows enabled True -> firewall enabled
        if isinstance(j, list):
            return any([p.get("Enabled") for p in j])
        if isinstance(j, dict):
            return j.get("Enabled", False)
        return None
    except Exception:
        return None

def post_heartbeat():
    hb = {
        "host_id": HOST_ID,
        "hostname": platform.node(),
        "os_version": platform.platform(),
        "defender_enabled": get_defender_status(),
        "firewall_enabled": get_firewall_status(),
        "timestamp": utcnow()
    }
    r = requests.post(f"{SERVER_URL}/heartbeat", json=hb, headers=HEADERS, timeout=6)
    return r.ok

def extract_pe_basic(path):
    """Return minimal static features (safe - doesn't execute file)"""
    try:
        pe = pefile.PE(str(path))
        sections = []
        for s in pe.sections:
            sections.append({
                "name": s.Name.decode(errors="ignore").strip('\x00'),
                "virtual_size": s.Misc_VirtualSize,
                "raw_size": s.SizeOfRawData,
                "entropy": round(s.get_entropy(),4)
            })
        imports = []
        try:
            for imp in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
                imports.append(imp.dll.decode(errors="ignore"))
        except Exception:
            imports = []
        sha256 = sha256_of_file(path)
        feats = {
            "host_id": HOST_ID,
            "sha256": sha256,
            "file_path": str(path),
            "size": Path(path).stat().st_size,
            "num_sections": len(pe.sections),
            "sections": sections,
            "imports": imports,
            "entropy": sum([s["entropy"] for s in sections]) / len(sections) if sections else None,
            "timestamp": utcnow()
        }
        return feats
    except Exception as e:
        return {"error": str(e)}

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def get_recent_sysmon_events(max_events=50):
    """
    Pull recent Sysmon events (ProcessCreate and NetworkConnect) using PowerShell.
    This returns simplified events as a list of dicts.
    """
    ps_script = r"""
    $events = Get-WinEvent -LogName "Microsoft-Windows-Sysmon/Operational" -MaxEvents {max_events} |
      ForEach-Object {
        $xml = [xml]$_.ToXml()
        $e = @{{}}
        $e.EventID = $xml.Event.System.EventID
        $e.TimeCreated = $xml.Event.System.TimeCreated.SystemTime
        # flatten properties
        $props = $xml.Event.EventData.Data
        $i = 0
        foreach ($p in $props) {{
            $e["p$i"] = $p.'#text'
            $i++
        }}
        $e
      }
    $events | ConvertTo-Json -Depth 4
    """.format(max_events=max_events)
    try:
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10, text=True)
        if not out.strip():
            return []
        raw = json.loads(out)
        # Normalize into simple semantic events if possible
        normalized = []
        for r in (raw if isinstance(raw, list) else [raw]):
            ev = {}
            ev["EventID"] = int(r.get("EventID", 0))
            ev["TimeCreated"] = r.get("TimeCreated")
            if ev["EventID"] == 1:  # ProcessCreate
                # Sysmon v13 has fixed ordering; we map best-effort
                # p0..pN mapping may vary by Sysmon version; do best-effort pick
                ev["EventType"] = "ProcessCreate"
                ev["Process"] = r.get("p4") or r.get("p0") or ""
                ev["CommandLine"] = r.get("p5") or r.get("p1") or ""
            elif ev["EventID"] in (3, 22):  # NetworkConnect / DNS
                ev["EventType"] = "NetworkConnect"
                ev["Source"] = r.get("p3") or ""
                ev["Destination"] = r.get("p4") or r.get("p2") or ""
            else:
                ev["EventType"] = f"Sysmon{ev['EventID']}"
            normalized.append(ev)
        return normalized
    except Exception:
        return []

def push_events(events):
    if not events:
        return True
    payload = {"host_id": HOST_ID, "events": events}
    r = requests.post(f"{SERVER_URL}/events", json=payload, headers=HEADERS, timeout=8)
    return r.ok

def push_features(feat):
    r = requests.post(f"{SERVER_URL}/features", json=feat, headers=HEADERS, timeout=8)
    return r.ok

def scan_sample_and_send(path):
    feats = extract_pe_basic(path)
    if "error" not in feats:
        return push_features(feats)
    return False

def main_loop(poll_secs=60):
    print("Agent starting, host id:", HOST_ID)
    # initial heartbeat
    try:
        post_heartbeat()
    except Exception as e:
        print("Unable to contact server:", e)
    while True:
        try:
            ok = post_heartbeat()
            if not ok:
                print("Heartbeat failed")
        except Exception as e:
            print("Heartbeat exception:", e)

        # gather recent sysmon events and push
        try:
            events = get_recent_sysmon_events(max_events=50)
            if events:
                push_events(events)
        except Exception as e:
            print("events error", e)

        # optional: scan files placed into a watch folder (example)
        watch_dir = Path("./to_scan")
        watch_dir.mkdir(exist_ok=True)
        for f in watch_dir.glob("*"):
            try:
                if f.is_file():
                    print("Scanning sample:", f.name)
                    scanned = scan_sample_and_send(str(f))
                    if scanned:
                        # move to scanned folder
                        dest = Path("./scanned")
                        dest.mkdir(exist_ok=True)
                        f.rename(dest / f.name)
            except Exception as e:
                print("scan exception", e)

        time.sleep(poll_secs)

if __name__ == "__main__":
    main_loop(poll_secs=60)
