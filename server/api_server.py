#!/usr/bin/env python3
"""
FastAPI Server for Malware Detection
This API accepts PE file uploads, extracts metadata features, and uses the trained
Random Forest model to predict if the file is malware.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import timedelta
from pymongo import MongoClient
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import os
import secrets
import tempfile
import json
from datetime import datetime
import numpy as np
import requests
from urllib.parse import urlencode
from pathlib import Path

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    load_dotenv()
else:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

from feature_extractor import PEFeatureExtractor
from malware_classifier import MalwareTypeClassifier
from adaptive_learning import AdaptiveLearningManager
from performance_monitor import PerformanceMonitor

# Import sandbox client and LSTM detector
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sandbox'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))

try:
    from windows_sandbox_client import WindowsSandboxClient  # type: ignore
except ImportError:
    WindowsSandboxClient = None
    print("⚠️ Warning: WindowsSandboxClient not available")

try:
    from predict_lstm_behavioral import BehavioralMalwareDetector  # type: ignore
except ImportError:
    BehavioralMalwareDetector = None
    print("⚠️ Warning: BehavioralMalwareDetector not available")

# ========== CONFIG ==========
MODEL_DIR = r"C:\Users\Admin\github-classroom\Caty175\poly_trial\Model"
COMPONENTS_DIR = os.path.join(MODEL_DIR, "components")
MODEL_PATH = os.path.join(COMPONENTS_DIR, "randomforest_malware_detector.pkl")
FEATURE_LIST_PATH = os.path.join(COMPONENTS_DIR, "feature_list.json")
LSTM_METADATA_PATH = os.path.join(COMPONENTS_DIR, "lstm_model_metadata.json")
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Sandbox configuration
# To configure VM IP address, either:
# 1. Set environment variable: export SANDBOX_VM_IP=192.168.1.100
# 2. Or change the default value below directly
SANDBOX_VM_IP = os.getenv("SANDBOX_VM_IP", "192.168.1.100")  # Change this to your VM's IP address
SANDBOX_VM_PORT = int(os.getenv("SANDBOX_VM_PORT", "5000"))
SANDBOX_ANALYSIS_DURATION = int(os.getenv("SANDBOX_ANALYSIS_DURATION", "120"))
RF_MALWARE_THRESHOLD = 0.60  # 60% threshold to trigger sandbox analysis
RF_WEIGHT = 0.30  # 30% weight for Random Forest
LSTM_WEIGHT = 0.70  # 70% weight for LSTM
# ============================


# Initialize FastAPI app
app = FastAPI(
    title="Polymorphic Malware Detection API",
    description="""
    ## 🦠 Polymorphic Malware Detection API
    Welcome to the **Polymorphic Malware Detection API**! This API allows you to:
    - Upload PE files and scan for malware
    - Get detailed feature analysis
    - View model information and health status
    
    ### Features
    - 🚀 Fast and accurate malware detection
    - 📊 Detailed feature importance
    - 🛡️ Health and model info endpoints
    
    For more details, see the [README](../README.md).
    """,
    version="1.0.0",
    contact={
        "name": "Caty175",
        "url": "https://github.com/Caty175/poly_trial1",
        "email": "caty175@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Mount static and template directories
app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")

client = MongoClient("mongodb://localhost:27017/")
db = client["poly_trial"]
users_collection = db["users"]
scan_results_collection = db["scan_results"]
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Add session middleware for user session management (after SECRET_KEY is defined)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
GOOGLE_SCOPES = "openid email profile"

google_oauth_states: Dict[str, Dict[str, str]] = {}

# ------------------ AUTH UI ROUTES ------------------
@app.get("/login", response_class=HTMLResponse, tags=["Auth"])
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse, tags=["Auth"])
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/profile", response_class=HTMLResponse, tags=["Profile"])
def profile_page(request: Request):
    return templates.TemplateResponse("profile.html", {"request": request})


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def chrome_devtools_probe():
    return JSONResponse({"status": "ok"})


# ------------------ AUTH API ENDPOINTS ------------------
def verify_password(plain_password, hashed_password):
    """Verify a password against its hash using pbkdf2_sha256."""
    if not isinstance(plain_password, str):
        plain_password = str(plain_password)
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password using pbkdf2_sha256 (no length limit, no bcrypt compatibility issues)."""
    if not isinstance(password, str):
        password = str(password)
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user_email(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization")
) -> Optional[str]:
    """
    Extract user email from session (preferred) or JWT token in Authorization header.
    Returns None if both are missing or invalid (allows unauthenticated access).
    """
    # First, try to get from session
    session_email = request.session.get("user_email")
    if session_email:
        return session_email
    
    # Fallback to JWT token
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>" format
        parts = authorization.split()
        if len(parts) == 2:
            scheme, token = parts
            if scheme.lower() != "bearer":
                return None
        elif len(parts) == 1:
            # Try to use authorization as token directly (for compatibility)
            token = authorization
        else:
            return None
    except (ValueError, AttributeError):
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        return user_email
    except JWTError:
        return None


def get_current_user_id(
    request: Request
) -> Optional[str]:
    """
    Extract user ID from session.
    Returns None if not in session (allows unauthenticated access).
    """
    return request.session.get("user_id")


def require_google_oauth_config():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth is not configured. Define GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."
        )


def build_google_oauth_url(flow: str) -> str:
    require_google_oauth_config()
    state_token = secrets.token_urlsafe(24)
    google_oauth_states[state_token] = {
        "flow": flow,
        "created_at": datetime.utcnow().isoformat()
    }
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "state": state_token,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


@app.get("/auth/login/google", tags=["Auth"])
def google_login_flow():
    auth_url = build_google_oauth_url("login")
    return RedirectResponse(auth_url, status_code=302)


@app.get("/auth/signup/google", tags=["Auth"])
def google_signup_flow():
    auth_url = build_google_oauth_url("signup")
    return RedirectResponse(auth_url, status_code=302)


@app.get("/auth/google/callback", response_class=HTMLResponse, tags=["Auth"])
def google_oauth_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        raise HTTPException(status_code=400, detail=f"Google authentication failed: {error}")

    if not code or not state:
        raise HTTPException(status_code=400, detail="Invalid Google authentication response.")

    state_entry = google_oauth_states.pop(state, None)
    if not state_entry:
        raise HTTPException(status_code=400, detail="Invalid or expired Google authentication state.")

    flow = state_entry.get("flow", "login")

    require_google_oauth_config()

    token_response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )

    if token_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to exchange Google authorization code.")

    token_payload = token_response.json()
    access_token = token_payload.get("access_token")

    if not access_token:
        raise HTTPException(status_code=400, detail="Google token exchange did not return an access token.")

    userinfo_response = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )

    if userinfo_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to retrieve Google account information.")

    userinfo = userinfo_response.json()
    email = userinfo.get("email")
    if not email or not userinfo.get("email_verified", False):
        raise HTTPException(status_code=400, detail="Google account email is not verified.")

    first_name = userinfo.get("given_name", "")
    last_name = userinfo.get("family_name", "")

    user = users_collection.find_one({"work_email": email})

    if not user:
        user_doc = {
            "first_name": first_name,
            "last_name": last_name,
            "work_email": email,
            "institution": userinfo.get("hd") or "Google Workspace",
            "password_hash": None,
            "auth_provider": "google",
        }
        result = users_collection.insert_one(user_doc)
        # Get the newly created user
        user = users_collection.find_one({"_id": result.inserted_id})
    else:
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {
                "first_name": first_name or user.get("first_name"),
                "last_name": last_name or user.get("last_name"),
                "auth_provider": "google",
            }}
        )

    jwt_token = create_access_token({"sub": email})
    
    # Store user email and ID in session
    request.session["user_email"] = email
    if user and "_id" in user:
        request.session["user_id"] = str(user["_id"])

    target_redirect = "/dashboard"
    script = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Google Authentication • Redirecting</title>
    </head>
    <body style="margin:0;font-family:Arial,sans-serif;background:#050005;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;">
        <div style="text-align:center;">
            <h1>Authentication successful</h1>
            <p>Redirecting you to your dashboard…</p>
        </div>
        <script>
            try {{
                window.localStorage.setItem('access_token', {json.dumps(jwt_token)});
            }} catch (err) {{
                console.error('Unable to store access token', err);
            }}
            window.location.href = {json.dumps(target_redirect)};
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=script)

@app.post("/api/signup", tags=["Auth"])
async def signup(payload: dict):
    required_fields = ["first_name", "last_name", "work_email", "institution", "password"]
    for field in required_fields:
        if not payload.get(field):
            return JSONResponse(status_code=400, content={"detail": f"{field.replace('_', ' ').title()} is required."})
    work_email = payload["work_email"]
    if users_collection.find_one({"work_email": work_email}):
        return JSONResponse(status_code=400, content={"detail": "Work email already registered."})
    user_doc = {
        "first_name": payload["first_name"],
        "last_name": payload["last_name"],
        "work_email": work_email,
        "institution": payload["institution"],
        "password_hash": get_password_hash(payload["password"])
    }
    users_collection.insert_one(user_doc)
    return {"success": True}

@app.post("/api/login", tags=["Auth"])
async def login(request: Request, payload: dict):
    work_email = payload.get("work_email")
    password = payload.get("password")
    user = users_collection.find_one({"work_email": work_email})
    password_hash = user.get("password_hash") if user else None
    if not user or not password_hash or not verify_password(password, password_hash):
        return JSONResponse(status_code=401, content={"detail": "Invalid email or password."})
    
    # Create JWT token
    access_token = create_access_token({"sub": work_email})
    
    # Store user email in session
    request.session["user_email"] = work_email
    request.session["user_id"] = str(user["_id"])
    
    return {"access_token": access_token, "token_type": "bearer", "user_email": work_email}

@app.get("/api/profile", tags=["Profile"])
async def get_profile(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Get current user's profile information."""
    user_email = get_current_user_email(request, authorization)
    
    if not user_email:
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required"}
        )
    
    user = users_collection.find_one({"work_email": user_email})
    
    if not user:
        return JSONResponse(
            status_code=404,
            content={"detail": "User not found"}
        )
    
    # Return user profile (exclude sensitive data)
    profile_data = {
        "work_email": user.get("work_email"),
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "institution": user.get("institution"),
        "auth_provider": user.get("auth_provider", "email")
    }
    
    return profile_data

# Global variables for model and feature extractor
model = None
feature_extractor = None
feature_metadata = None
lstm_detector = None
lstm_metadata = None
malware_classifier = None
adaptive_manager = None
performance_monitor = None


class PredictionResponse(BaseModel):
    """Response model for malware prediction."""
    filename: str
    prediction: str  # "malware" or "benign"
    confidence: float
    malware_probability: float
    benign_probability: float
    timestamp: str
    features_extracted: int
    file_size: int
    random_forest_probability: Optional[float] = None
    lstm_probability: Optional[float] = None
    sandbox_analysis_performed: bool = False
    malware_type: Optional[str] = None
    malware_classification: Optional[Dict[str, Any]] = None
    
    class Config:
        # Ensure None values are included in JSON response
        exclude_none = False


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    feature_extractor_loaded: bool
    num_features: int
    model_info: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def load_model():
    """Load the trained model and feature extractor on startup."""
    global model, feature_extractor, feature_metadata, lstm_detector, lstm_metadata, malware_classifier

    try:
        # Load Random Forest model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        model = joblib.load(MODEL_PATH)
        print(f"✅ Random Forest model loaded from {MODEL_PATH}")

        # Load feature metadata
        if not os.path.exists(FEATURE_LIST_PATH):
            raise FileNotFoundError(f"Feature list not found at {FEATURE_LIST_PATH}")

        with open(FEATURE_LIST_PATH, 'r') as f:
            feature_metadata = json.load(f)

        # Initialize feature extractor
        feature_extractor = PEFeatureExtractor(FEATURE_LIST_PATH)
        print(f"✅ Loaded {len(feature_metadata['features'])} features from {FEATURE_LIST_PATH}")
        print(f"✅ Feature extractor initialized with {len(feature_metadata['features'])} features")

        # Load LSTM model if available
        if BehavioralMalwareDetector is not None:
            try:
                lstm_detector = BehavioralMalwareDetector()
                with open(LSTM_METADATA_PATH, 'r') as f:
                    lstm_metadata = json.load(f)
                print(f"✅ LSTM behavioral detector loaded")
            except Exception as e:
                print(f"⚠️ Warning: Could not load LSTM model: {e}")
                lstm_detector = None
        else:
            print("⚠️ Warning: LSTM detector not available (module not imported)")

        # Initialize malware classifier
        try:
            malware_classifier = MalwareTypeClassifier()
            print(f"✅ Malware type classifier initialized")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize malware classifier: {e}")
            malware_classifier = None

        # Initialize adaptive learning components
        try:
            global adaptive_manager, performance_monitor
            adaptive_manager = AdaptiveLearningManager()
            performance_monitor = PerformanceMonitor()
            print(f"✅ Adaptive learning manager initialized")
            print(f"✅ Performance monitor initialized")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize adaptive learning: {e}")
            adaptive_manager = None
            performance_monitor = None

        print("🚀 API server ready!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def convert_behavioral_report_to_lstm_format(behavioral_report: dict, lstm_metadata: dict) -> dict:
    """
    Convert behavioral report JSON from sandbox to LSTM model format.
    
    Args:
        behavioral_report: JSON report from Windows sandbox
        lstm_metadata: LSTM model metadata with feature names
        
    Returns:
        dict: Features in LSTM format (API features + static features)
    """
    # Get feature lists from metadata
    api_features = lstm_metadata['feature_metadata']['api_features']
    static_features = lstm_metadata['feature_metadata']['static_features']
    
    # Initialize feature dict with zeros
    features = {}
    for feat in api_features + static_features:
        features[feat] = 0
    
    # Extract API calls from report
    if 'api_calls' in behavioral_report:
        api_calls = behavioral_report['api_calls']
        for api_name, count in api_calls.items():
            # Match API name (with or without API_ prefix)
            api_key = api_name if api_name.startswith('API_') else f'API_{api_name}'
            if api_key in features:
                features[api_key] = int(count)
    
    # Extract file operations
    if 'file_operations' in behavioral_report:
        file_ops = behavioral_report['file_operations']
        for op_name in ['file_created', 'file_deleted', 'file_read', 'file_written']:
            if op_name in file_ops and op_name in features:
                features[op_name] = int(file_ops[op_name])
    
    # Extract DLL loading frequencies
    if 'dll_loaded' in behavioral_report:
        dll_loaded = behavioral_report['dll_loaded']
        for dll_name, count in dll_loaded.items():
            # Normalize DLL name (lowercase, handle paths)
            dll_key = f"dll_freq_{dll_name.lower()}"
            if dll_key in features:
                features[dll_key] = int(count)
            # Also try with full path format
            dll_key_path = f"dll_freq_{dll_name}"
            if dll_key_path in features:
                features[dll_key_path] = int(count)
    
    # Extract behavioral indicators
    if 'behavioral_indicators' in behavioral_report:
        indicators = behavioral_report['behavioral_indicators']
        for indicator_name in ['regkey_read', 'directory_enumerated', 'dll_loaded_count', 'resolves_host']:
            if indicator_name in indicators and indicator_name in features:
                features[indicator_name] = int(indicators[indicator_name])
    
    # Handle command_line if present
    if 'command_line' in behavioral_report and 'command_line' in features:
        features['command_line'] = 1 if behavioral_report['command_line'] else 0
    
    return features


@app.get(
    "/",
    response_model=HealthResponse,
    summary="API Health Check",
    description="""
    ### Health Check
    Returns the health status of the API, model, and feature extractor.
    """,
    response_description="Health status and model info.",
    tags=["Health"]
)
async def root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "feature_extractor_loaded": feature_extractor is not None,
        "num_features": len(feature_metadata['features']) if feature_metadata else 0,
        "model_info": {
            "training_date": feature_metadata.get('training_date', 'unknown') if feature_metadata else 'unknown',
            "accuracy": feature_metadata.get('metrics', {}).get('accuracy', 'unknown') if feature_metadata else 'unknown',
            "roc_auc": feature_metadata.get('metrics', {}).get('roc_auc', 'unknown') if feature_metadata else 'unknown'
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check (Alias)",
    description="Alias for the root endpoint. Returns health status, model info, and feature extractor status.",
    response_description="Health status and model info.",
    tags=["Health"]
)
async def health_check():
    return await root()


@app.post(
    "/scan",
    response_model=PredictionResponse,
    summary="Scan PE File for Malware",
    description="""
    ### Scan PE File
    Upload a PE file to scan for malware. Returns prediction, confidence, and probability scores.
    """,
    response_description="Prediction result with confidence scores.",
    tags=["Scan"],
    responses={
        200: {
            "description": "Successful scan result",
            "content": {
                "application/json": {
                    "example": {
                        "filename": "sample.exe",
                        "prediction": "malware",
                        "confidence": 0.98,
                        "malware_probability": 0.98,
                        "benign_probability": 0.02,
                        "timestamp": "2025-11-12T12:34:56",
                        "features_extracted": 120,
                        "file_size": 123456
                    }
                }
            }
        },
        400: {"description": "Invalid file or extraction error."},
        503: {"description": "Model not loaded."}
    }
)
async def scan_file(
    request: Request,
    file: UploadFile = File(..., description="PE file to scan for malware"),
    user_email: Optional[str] = Depends(get_current_user_email)
):
    if model is None or feature_extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Read file content
    try:
        file_content = await file.read()
        file_size = len(file_content)
        
        # Check file size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f} MB"
            )
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Save to temporary file for processing
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Extract features
        try:
            features_array = feature_extractor.extract_features_as_array(temp_file_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting features. File may not be a valid PE file: {str(e)}"
            )
        
        # Make Random Forest prediction
        try:
            prediction_proba = model.predict_proba(features_array)[0]
            prediction_class = model.predict(features_array)[0]
            
            rf_benign_prob = float(prediction_proba[0])
            rf_malware_prob = float(prediction_proba[1])
            rf_confidence = max(rf_benign_prob, rf_malware_prob)
            
            rf_prediction_label = "malware" if prediction_class == 1 else "benign"
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
        
        # Initialize response data with RF results
        # When LSTM is not triggered, apply RF weight (30%) to final probability
        # But keep prediction based on RF result itself
        weighted_malware_prob = RF_WEIGHT * rf_malware_prob
        weighted_benign_prob = RF_WEIGHT * rf_benign_prob
        weighted_confidence = max(weighted_malware_prob, weighted_benign_prob)
        # Prediction is still based on RF result, not weighted result
        # (since weighted result is only 30% of ensemble, threshold doesn't apply)
        
        # Initialize lstm_features for adaptive learning (may be set later)
        lstm_features = None
        
        response_data = {
            "filename": file.filename,
            "prediction": rf_prediction_label,
            "confidence": weighted_confidence,
            "malware_probability": weighted_malware_prob,
            "benign_probability": weighted_benign_prob,
            "timestamp": datetime.now().isoformat(),
            "features_extracted": features_array.shape[1],
            "file_size": file_size,
            "random_forest_probability": rf_malware_prob,
            "lstm_probability": None,
            "sandbox_analysis_performed": False
        }
        
        # Check if RF malware probability > 60% threshold
        # If yes, trigger sandbox analysis and LSTM prediction
        if rf_malware_prob >= RF_MALWARE_THRESHOLD and WindowsSandboxClient is not None and lstm_detector is not None and lstm_metadata is not None:
            try:
                print(f"🔍 RF malware probability ({rf_malware_prob:.2%}) >= threshold ({RF_MALWARE_THRESHOLD:.2%})")
                print(f"📤 Triggering sandbox analysis and LSTM prediction...")
                
                # Create sandbox client
                sandbox_client = WindowsSandboxClient(SANDBOX_VM_IP, SANDBOX_VM_PORT)
                
                # Send file to sandbox for analysis
                print(f"📤 Sending file to sandbox server at {SANDBOX_VM_IP}:{SANDBOX_VM_PORT}...")
                sandbox_result = sandbox_client.analyze_file(
                    temp_file_path,
                    duration=SANDBOX_ANALYSIS_DURATION,
                    save_report=False
                )
                
                if sandbox_result and 'report' in sandbox_result:
                    behavioral_report = sandbox_result['report']
                    print(f"✅ Sandbox analysis complete")
                    
                    # Convert behavioral report to LSTM format
                    print(f"🔄 Converting behavioral report to LSTM format...")
                    lstm_features = convert_behavioral_report_to_lstm_format(behavioral_report, lstm_metadata)
                    
                    # Run LSTM prediction
                    print(f"🤖 Running LSTM prediction...")
                    lstm_result = lstm_detector.predict(lstm_features)
                    
                    lstm_malware_prob = lstm_result['malware_probability']
                    lstm_benign_prob = lstm_result['benign_probability']
                    
                    print(f"✅ LSTM prediction: {lstm_result['prediction']} (confidence: {lstm_result['confidence']:.2%})")
                    
                    # Combine RF (30%) and LSTM (70%) probabilities
                    combined_malware_prob = (RF_WEIGHT * rf_malware_prob) + (LSTM_WEIGHT * lstm_malware_prob)
                    combined_benign_prob = (RF_WEIGHT * rf_benign_prob) + (LSTM_WEIGHT * lstm_benign_prob)
                    combined_confidence = max(combined_malware_prob, combined_benign_prob)
                    combined_prediction = "malware" if combined_malware_prob >= 0.5 else "benign"
                    
                    print(f"📊 Combined prediction: {combined_prediction}")
                    print(f"   RF (30%): {rf_malware_prob:.2%}")
                    print(f"   LSTM (70%): {lstm_malware_prob:.2%}")
                    print(f"   Combined: {combined_malware_prob:.2%}")
                    
                    # Classify malware type if malware detected
                    malware_classification_result = None
                    if combined_prediction == "malware" and malware_classifier is not None:
                        try:
                            print(f"🔍 Classifying malware type based on behavioral features...")
                            malware_classification_result = malware_classifier.classify_malware_type(lstm_features)
                            print(f"✅ Malware type: {malware_classification_result['malware_type']}")
                        except Exception as e:
                            print(f"⚠️ Warning: Could not classify malware type: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Update response with combined results
                    response_data.update({
                        "prediction": combined_prediction,
                        "confidence": combined_confidence,
                        "malware_probability": combined_malware_prob,
                        "benign_probability": combined_benign_prob,
                        "random_forest_probability": rf_malware_prob,
                        "lstm_probability": lstm_malware_prob,
                        "sandbox_analysis_performed": True,
                        "malware_type": malware_classification_result['malware_type'] if malware_classification_result else None,
                        "malware_classification": malware_classification_result if malware_classification_result else None
                    })
                else:
                    print(f"⚠️ Sandbox analysis failed or returned no report")
                    # Continue with RF-only results
                    
            except Exception as e:
                print(f"⚠️ Error during sandbox/LSTM analysis: {e}")
                import traceback
                traceback.print_exc()
                # Continue with RF-only results if sandbox/LSTM fails
        else:
            if rf_malware_prob < RF_MALWARE_THRESHOLD:
                print(f"ℹ️ RF malware probability ({rf_malware_prob:.2%}) < threshold ({RF_MALWARE_THRESHOLD:.2%}), skipping sandbox analysis")
            elif WindowsSandboxClient is None:
                print(f"⚠️ WindowsSandboxClient not available, skipping sandbox analysis")
            elif lstm_detector is None:
                print(f"⚠️ LSTM detector not available, skipping sandbox analysis")

        # Save result to database
        try:
            # Resolve user_email and user_id from dependency/session
            resolved_user_email = user_email if isinstance(user_email, str) else None
            # Get user_id from request session
            resolved_user_id = request.session.get("user_id")
            
            scan_result_doc = {
                "filename": response_data["filename"],
                "prediction": response_data["prediction"],
                "confidence": response_data["confidence"],
                "malware_probability": response_data["malware_probability"],
                "benign_probability": response_data["benign_probability"],
                "timestamp": response_data["timestamp"],
                "features_extracted": response_data["features_extracted"],
                "file_size": response_data["file_size"],
                "random_forest_probability": response_data.get("random_forest_probability"),
                "lstm_probability": response_data.get("lstm_probability"),
                "sandbox_analysis_performed": response_data.get("sandbox_analysis_performed", False),
                "user_email": resolved_user_email,  # Keep for backward compatibility
                "user_id": resolved_user_id,  # Primary identifier for user association
                "created_at": datetime.utcnow()
            }
            result = scan_results_collection.insert_one(scan_result_doc)
            print(f"💾 Scan result saved to database with ID: {result.inserted_id}" + (f" (user_id: {resolved_user_id})" if resolved_user_id else " (anonymous)"))
        except Exception as e:
            print(f"⚠️ Warning: Failed to save scan result to database: {e}")
            import traceback
            traceback.print_exc()
            # Continue even if database save fails
        
        # Calculate file hash before cleanup (needed for adaptive learning)
        file_hash = None
        if adaptive_manager is not None:
            try:
                file_hash = adaptive_manager.calculate_file_hash(file_content)
            except Exception as e:
                print(f"⚠️ Warning: Failed to calculate file hash: {e}")
        
        response = PredictionResponse(**response_data)
        
        # Add sample to adaptive learning system (after response created, before cleanup)
        if adaptive_manager is not None and file_hash:
            try:
                # Extract features dict for storage (if temp file still exists)
                features_dict = {}
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        features_dict = feature_extractor.extract_features(temp_file_path)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not extract features for adaptive learning: {e}")
                
                # Get behavioral features if available (from LSTM prediction)
                behavioral_features = None
                if response_data.get("sandbox_analysis_performed") and lstm_features:
                    # Use the LSTM features that were already extracted
                    behavioral_features = lstm_features
                
                adaptive_manager.add_sample(
                    file_hash=file_hash,
                    features=features_dict,
                    prediction=response_data["prediction"],
                    confidence=response_data["confidence"],
                    rf_probability=response_data.get("random_forest_probability"),
                    lstm_probability=response_data.get("lstm_probability"),
                    behavioral_features=behavioral_features,
                    user_id=resolved_user_id,
                    filename=file.filename
                )
                
                # Record prediction for performance monitoring
                if performance_monitor is not None:
                    performance_monitor.record_prediction(
                        file_hash=file_hash,
                        prediction=response_data["prediction"],
                        confidence=response_data["confidence"],
                        rf_probability=response_data.get("random_forest_probability"),
                        lstm_probability=response_data.get("lstm_probability")
                    )
            except Exception as e:
                print(f"⚠️ Warning: Failed to add sample to adaptive learning: {e}")
                import traceback
                traceback.print_exc()
        
        return response
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"⚠️ Error deleting temporary file: {e}")


@app.post(
    "/scan/detailed",
    summary="Scan PE File (Detailed)",
    description="""
    ### Scan PE File (Detailed)
    Upload a PE file for malware scan with detailed feature information. Returns prediction, top features, and all extracted features.
    """,
    response_description="Detailed prediction result with features.",
    tags=["Scan"],
    responses={
        200: {
            "description": "Detailed scan result",
            "content": {
                "application/json": {
                    "example": {
                        "filename": "sample.exe",
                        "prediction": "benign",
                        "confidence": 0.95,
                        "malware_probability": 0.05,
                        "benign_probability": 0.95,
                        "timestamp": "2025-11-12T12:34:56",
                        "features_extracted": 120,
                        "file_size": 123456,
                        "top_features": [
                            {"feature": "SizeOfCode", "value": 4096, "importance": 0.12},
                            {"feature": "Imports", "value": 15, "importance": 0.09}
                        ],
                        "all_features": {"SizeOfCode": 4096, "Imports": 15, "Entropy": 7.2}
                    }
                }
            }
        },
        400: {"description": "Invalid file or extraction error."},
        503: {"description": "Model not loaded."}
    }
)
async def scan_file_detailed(
    request: Request,
    file: UploadFile = File(..., description="PE file to scan for malware (detailed)"),
    user_email: Optional[str] = Depends(get_current_user_email)
):
    # Get basic prediction (scan_file will use the same user_email from dependency)
    basic_result = await scan_file(request, file)
    
    # Re-read file for feature extraction (since file was already consumed)
    await file.seek(0)
    file_content = await file.read()
    
    # Save to temporary file again
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Extract features as dictionary
        features_dict = feature_extractor.extract_features(temp_file_path)
        
        # Get top important features from metadata
        top_features = feature_metadata.get('top_features', [])[:10] if feature_metadata else []
        
        # Get values for top features
        top_feature_values = []
        for feat_info in top_features:
            feat_name = feat_info['feature']
            feat_value = features_dict.get(feat_name, 0)
            top_feature_values.append({
                'feature': feat_name,
                'value': feat_value,
                'importance': feat_info['importance']
            })
        
        # Combine with basic result
        # Use exclude_none=False to ensure all fields are included in response
        detailed_result = basic_result.dict(exclude_none=False)
        detailed_result['top_features'] = top_feature_values
        detailed_result['all_features'] = features_dict
        
        return JSONResponse(content=detailed_result)
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"⚠️ Error deleting temporary file: {e}")


@app.get(
    "/model/info",
    summary="Get Model Information",
    description="""
    ### Model Information
    Get information about the loaded model, including number of features, training date, parameters, metrics, and top features.
    """,
    response_description="Model information and metadata.",
    tags=["Model Info"],
    responses={
        200: {
            "description": "Model information and metadata",
            "content": {
                "application/json": {
                    "example": {
                        "num_features": 120,
                        "training_date": "2025-10-01",
                        "model_params": {"n_estimators": 100, "max_depth": 10},
                        "metrics": {"accuracy": 0.97, "roc_auc": 0.99},
                        "top_features": [
                            {"feature": "SizeOfCode", "importance": 0.12},
                            {"feature": "Imports", "importance": 0.09}
                        ]
                    }
                }
            }
        },
        503: {"description": "Model metadata not loaded."}
    }
)
async def get_model_info():
    if feature_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")

    return {
        "num_features": feature_metadata.get('num_features', 0),
        "training_date": feature_metadata.get('training_date', 'unknown'),
        "model_params": feature_metadata.get('model_params', {}),
        "metrics": feature_metadata.get('metrics', {}),
        "top_features": feature_metadata.get('top_features', [])[:20]
    }


@app.get(
    "/scan/history",
    summary="Get Scan History",
    description="""
    ### Scan History
    Retrieve past scan results from the database for the authenticated user. Results are sorted by most recent first.
    
    **Authentication Required**: This endpoint requires a valid JWT token in the Authorization header.
    
    Query Parameters:
    - `limit`: Maximum number of results to return (default: 50, max: 500)
    - `skip`: Number of results to skip for pagination (default: 0)
    - `prediction`: Filter by prediction type ("malware" or "benign")
    """,
    response_description="List of past scan results for the authenticated user.",
    tags=["Scan"],
    responses={
        200: {
            "description": "List of scan results",
            "content": {
                "application/json": {
                    "example": {
                        "total": 100,
                        "limit": 50,
                        "skip": 0,
                        "results": [
                            {
                                "id": "507f1f77bcf86cd799439011",
                                "filename": "sample.exe",
                                "prediction": "malware",
                                "confidence": 0.98,
                                "malware_probability": 0.98,
                                "timestamp": "2025-11-12T12:34:56",
                                "file_size": 123456
                            }
                        ]
                    }
                }
            }
        },
        401: {"description": "Authentication required. Please provide a valid JWT token."}
    }
)
async def get_scan_history(
    request: Request,
    limit: int = 50,
    skip: int = 0,
    prediction: Optional[str] = None,
    user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Retrieve scan history from the database for the authenticated user.
    
    Args:
        request: FastAPI Request object to access session
        limit: Maximum number of results to return (1-500)
        skip: Number of results to skip for pagination
        prediction: Filter by prediction type ("malware" or "benign")
        user_id: User ID extracted from session (required)
    
    Returns:
        Dictionary with total count and list of scan results
    """
    # Require authentication - get user_id from session
    if not user_id:
        # Fallback: try to get from session directly
        user_id = request.session.get("user_id")
    
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please log in to view your scan history."
        )
    
    # Validate limit
    if limit < 1:
        limit = 50
    if limit > 500:
        limit = 500
    
    if skip < 0:
        skip = 0
    
    # Build query - filter by user_id (primary identifier from session)
    query = {"user_id": user_id}
    
    # Add prediction filter if specified
    if prediction and prediction.lower() in ["malware", "benign"]:
        query["prediction"] = prediction.lower()
    
    # Get total count
    total = scan_results_collection.count_documents(query)
    
    # Get results (sorted by most recent first)
    cursor = scan_results_collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
    
    results = []
    for doc in cursor:
        # Convert ObjectId to string and format result
        result = {
            "id": str(doc["_id"]),
            "filename": doc.get("filename", "unknown"),
            "prediction": doc.get("prediction", "unknown"),
            "confidence": doc.get("confidence", 0.0),
            "malware_probability": doc.get("malware_probability", 0.0),
            "benign_probability": doc.get("benign_probability", 0.0),
            "timestamp": doc.get("timestamp", ""),
            "file_size": doc.get("file_size", 0),
            "features_extracted": doc.get("features_extracted", 0),
            "random_forest_probability": doc.get("random_forest_probability"),
            "lstm_probability": doc.get("lstm_probability"),
            "sandbox_analysis_performed": doc.get("sandbox_analysis_performed", False),
            "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None
        }
        results.append(result)
    
    return {
        "total": total,
        "limit": limit,
        "skip": skip,
        "results": results
    }


@app.post(
    "/api/feedback",
    summary="Submit Feedback on Prediction",
    description="""
    ### Submit Feedback
    Submit feedback on a prediction to help improve the model. This feedback is used for adaptive learning.
    
    **Parameters:**
    - `file_hash`: SHA256 hash of the file (from scan result)
    - `prediction_was_correct`: Boolean indicating if prediction was correct
    - `actual_label`: Actual label ("malware" or "benign") if known
    - `comment`: Optional comment
    """,
    tags=["Adaptive Learning"],
    responses={
        200: {"description": "Feedback recorded successfully"},
        400: {"description": "Invalid input"},
        503: {"description": "Adaptive learning not available"}
    }
)
async def submit_feedback(
    request: Request,
    file_hash: str,
    prediction_was_correct: bool,
    actual_label: Optional[str] = None,
    comment: Optional[str] = None,
    user_id: Optional[str] = Depends(get_current_user_id)
):
    """Submit feedback on a prediction."""
    if adaptive_manager is None:
        raise HTTPException(status_code=503, detail="Adaptive learning not available")
    
    # Validate actual_label if provided
    if actual_label and actual_label not in ["malware", "benign"]:
        raise HTTPException(status_code=400, detail="actual_label must be 'malware' or 'benign'")
    
    try:
        adaptive_manager.add_feedback(
            file_hash=file_hash,
            prediction_was_correct=prediction_was_correct,
            actual_label=actual_label,
            user_id=user_id,
            comment=comment
        )
        
        return {
            "success": True,
            "message": "Feedback recorded successfully",
            "file_hash": file_hash
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")


@app.get(
    "/api/adaptive/statistics",
    summary="Get Adaptive Learning Statistics",
    description="""
    ### Adaptive Learning Statistics
    Get statistics about the adaptive learning system, including sample counts, feedback, and retraining status.
    """,
    tags=["Adaptive Learning"],
    responses={
        200: {"description": "Statistics retrieved successfully"},
        503: {"description": "Adaptive learning not available"}
    }
)
async def get_adaptive_statistics():
    """Get adaptive learning statistics."""
    if adaptive_manager is None:
        raise HTTPException(status_code=503, detail="Adaptive learning not available")
    
    try:
        stats = adaptive_manager.get_statistics()
        should_retrain, reason = adaptive_manager.should_trigger_retraining()
        
        return {
            **stats,
            "should_retrain": should_retrain,
            "retrain_reason": reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")


@app.get(
    "/api/adaptive/performance",
    summary="Get Performance Metrics",
    description="""
    ### Performance Metrics
    Get current performance metrics and drift detection results.
    """,
    tags=["Adaptive Learning"],
    responses={
        200: {"description": "Performance metrics retrieved successfully"},
        503: {"description": "Performance monitoring not available"}
    }
)
async def get_performance_metrics(days: int = 30):
    """Get performance metrics and drift detection."""
    if performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Get performance summary
        summary = performance_monitor.get_performance_summary(days=days)
        
        # Get drift detection results
        baseline_acc = 0.90  # Default baseline, could be loaded from model metadata
        drift_results = performance_monitor.check_all_drift_indicators(baseline_accuracy=baseline_acc)
        
        return {
            "summary": summary,
            "drift_detection": drift_results,
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving performance metrics: {str(e)}")


@app.get(
    "/api/adaptive/performance/trends",
    summary="Get Performance Trends",
    description="""
    ### Performance Trends
    Get performance metrics over time to visualize trends.
    """,
    tags=["Adaptive Learning"],
    responses={
        200: {"description": "Performance trends retrieved successfully"},
        503: {"description": "Adaptive learning not available"}
    }
)
async def get_performance_trends(days: int = 30):
    """Get performance trends over time."""
    if adaptive_manager is None:
        raise HTTPException(status_code=503, detail="Adaptive learning not available")
    
    try:
        trends = adaptive_manager.get_performance_trends(days=days)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving trends: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("POLYMORPHIC MALWARE DETECTION API")
    print("="*60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Feature list path: {FEATURE_LIST_PATH}")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

