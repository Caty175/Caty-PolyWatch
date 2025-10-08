"""Server backend package initializer.

This package provides backend utilities (e.g., MongoDB via `db.init_backend`).
It should NOT create a Flask app or register routes here to avoid circular
imports with the application package. The application is created in `app/`.
"""
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()

print("[DEBUG] Loaded .env successfully")
print("[DEBUG] BREVO_API_KEY:", bool(os.getenv("BREVO_API_KEY")))
print("[DEBUG] BREVO_SENDER_EMAIL:", os.getenv("BREVO_SENDER_EMAIL"))


# Re-export utilities for convenience if needed.
try:
    from .db import init_backend, mongo  # noqa: F401
except Exception:
    # Allow package import even if database dependencies are missing
    pass
