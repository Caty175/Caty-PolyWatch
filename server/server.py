
from fastapi import FastAPI
import uvicorn

# Import the enhanced FastAPI app from api_server.py
from api_server import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
