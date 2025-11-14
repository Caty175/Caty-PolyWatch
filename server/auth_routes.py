from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from passlib.context import CryptContext
from jose import jwt
from pymongo import MongoClient
from datetime import timedelta, datetime
from typing import Dict

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["poly_trial"]
users_collection = db["users"]

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Router setup
auth_router = APIRouter()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    if not isinstance(password, str):
        password = str(password)
    password_bytes = password.encode('utf-8')[:72]
    return pwd_context.hash(password_bytes.decode('utf-8', errors='ignore'))

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@auth_router.post("/signup", tags=["Auth"])
async def signup(payload: Dict):
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

@auth_router.post("/login", tags=["Auth"])
async def login(payload: Dict):
    work_email = payload.get("work_email")
    password = payload.get("password")
    user = users_collection.find_one({"work_email": work_email})
    if not user or not verify_password(password, user["password_hash"]):
        return JSONResponse(status_code=401, content={"detail": "Invalid email or password."})
    access_token = create_access_token({"sub": work_email})
    return {"access_token": access_token, "token_type": "bearer"}
