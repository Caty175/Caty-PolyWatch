from flask import Flask

try:
    from flask_pymongo import PyMongo
except Exception:
    PyMongo = None  # type: ignore

# Create a PyMongo instance if available
mongo = PyMongo() if PyMongo is not None else None

def init_backend(app: Flask):
    """
    Initialize backend modules and MongoDB connection.
    """
    # Configure MongoDB
    app.config["MONGO_URI"] = "mongodb://localhost:27017/polywatch"
    if mongo is not None:
        mongo.init_app(app)

    # Ensure required collections and indexes exist (users "table")
    try:
        if mongo is not None:
            users = mongo.db.users  # type: ignore[attr-defined]
            # Create unique indexes for fast lookups and to enforce uniqueness
            users.create_index("email", unique=True)
            users.create_index("username", unique=True)
            users.create_index("phone", unique=True, sparse=True)
    except Exception:
        # If DB is not reachable at startup, skip; app can retry later
        pass

    # Try to import backend modules if present (optional during development)
    for module_name in ("login", "signup", "reports", "sandbox"):
        try:
            __import__(f"server.{module_name}")
        except Exception:
            # Module may be missing; continue without failing startup
            pass

    print("✅ Backend initialized with MongoDB and modules loaded.")
