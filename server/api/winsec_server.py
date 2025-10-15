# winsec_server.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timezone
import hashlib
import joblib
import os

API_KEY = os.environ.get("WINSEC_API_KEY", "replace_with_secure_key")  # use env var in prod

app = FastAPI(title="WinSec Posture API - Prototype")

# In-memory stores (replace with real DB)
HOSTS: Dict[str, Dict] = {}
SAMPLES: Dict[str, Dict] = {}
EVENTS: Dict[str, List[Dict]] = {}
JOBS: Dict[str, Dict] = {}

class StaticFeatures(BaseModel):
    host_id: str
    sha256: str
    file_path: Optional[str] = None
    size: Optional[int] = None
    num_sections: Optional[int] = None
    sections: Optional[List[Dict]] = None
    imports: Optional[List[str]] = None
    entropy: Optional[float] = None
    timestamp: Optional[str] = None

class HostHeartbeat(BaseModel):
    host_id: str
    hostname: str
    os_version: str
    defender_enabled: Optional[bool] = None
    firewall_enabled: Optional[bool] = None
    timestamp: Optional[str] = None

class EventBatch(BaseModel):
    host_id: str
    events: List[Dict]  # expect normalized Sysmon/ETW-like events

def require_key(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def score_security(host_id: str):
    """
    Simple heuristic scoring function. Replace with ML model if available.
    Returns risk_score (0..1) and verdict.
    """
    host = HOSTS.get(host_id, {})
    events = EVENTS.get(host_id, [])
    samples = [s for s in SAMPLES.values() if s.get("host_id") == host_id]

    score = 0.0
    reasons = []

    # baseline: defender/firewall
    if host.get("defender_enabled") is False:
        score += 0.25
        reasons.append("Defender disabled")
    if host.get("firewall_enabled") is False:
        score += 0.2
        reasons.append("Firewall disabled")

    # static sample signals
    for s in samples:
        if s.get("entropy", 0) > 7.5:
            score += 0.15
            reasons.append(f"High entropy sample: {s.get('sha256')[:8]}")
        # suspicious import heuristics
        imports = s.get("imports") or []
        suspects = {"VirtualAlloc", "CreateRemoteThread", "InternetOpenUrl", "URLDownloadToFile"}
        if any(any(sus in imp for imp in imports) for sus in suspects):
            score += 0.15
            reasons.append(f"suspicious imports in sample {s.get('sha256')[:8]}")

    # behavioral events heuristics
    # Expect events to be simplified dictionaries with keys: EventType, Process, CommandLine, Target
    for e in events[-200:]:  # last 200 events
        typ = e.get("EventType","").lower()
        if typ in ("networkconnect","dns"):
            dst = e.get("Destination", "")
            # unusual domains or IPs - prototype: score if not local
            if dst and not dst.startswith(("192.","10.","172.")):
                score += 0.02
                reasons.append(f"net to {dst}")
        if typ == "processcreate":
            cmd = e.get("CommandLine","").lower()
            if "powershell" in cmd and ("-enc " in cmd or "invoke-expression" in cmd):
                score += 0.2
                reasons.append("suspicious encoded powershell observed")
            if cmd.count("\\") > 5 and ("temp" in cmd or "%temp%" in cmd):
                score += 0.05
                reasons.append("process launched from temp path")

    # clamp and normalize
    if score > 1.0:
        score = 1.0
    # verdict thresholds
    if score > 0.6:
        verdict = "HIGH"
    elif score > 0.25:
        verdict = "MEDIUM"
    else:
        verdict = "LOW"

    return {"risk_score": round(score, 3), "verdict": verdict, "reasons": reasons}

@app.post("/heartbeat")
def heartbeat(hb: HostHeartbeat, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    ts = hb.timestamp or datetime.now(timezone.utc).isoformat()
    HOSTS[hb.host_id] = {
        "host_id": hb.host_id,
        "hostname": hb.hostname,
        "os_version": hb.os_version,
        "defender_enabled": hb.defender_enabled,
        "firewall_enabled": hb.firewall_enabled,
        "last_seen": ts
    }
    return {"status": "ok", "host": HOSTS[hb.host_id]}

@app.post("/features")
def submit_features(feat: StaticFeatures, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    fid = hashlib.sha256((feat.sha256 + feat.host_id).encode()).hexdigest()
    SAMPLES[fid] = {**feat.dict(), "received_at": datetime.now(timezone.utc).isoformat(), "host_id": feat.host_id}
    return {"status": "ok", "sample_id": fid}

@app.post("/events")
def submit_events(batch: EventBatch, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    host_id = batch.host_id
    EVENTS.setdefault(host_id, []).extend(batch.events)
    # limit size in memory for prototype
    if len(EVENTS[host_id]) > 5000:
        EVENTS[host_id] = EVENTS[host_id][-5000:]
    return {"status": "ok", "events_received": len(batch.events)}

@app.get("/report/{host_id}")
def report(host_id: str, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    if host_id not in HOSTS:
        raise HTTPException(status_code=404, detail="host not found")
    score = score_security(host_id)
    return {
        "host": HOSTS[host_id],
        "latest_score": score,
        "samples_count": len([s for s in SAMPLES.values() if s.get("host_id")==host_id]),
        "recent_events": EVENTS.get(host_id, [])[-20:]
    }
