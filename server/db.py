from flask import Flask
from importlib import import_module
import logging
import os

try:
    from flask_pymongo import PyMongo
except ImportError as e:
    PyMongo = None
    logging.warning(f"PyMongo not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a PyMongo instance if available
mongo = PyMongo() if PyMongo is not None else None

def init_backend(app: Flask):
    """
    Initialize backend modules and MongoDB connection with proper error handling.
    """
    # Configure MongoDB with connection pooling and timeout settings
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/polywatch")
    app.config["MONGO_URI"] = mongo_uri
    app.config["MONGO_CONNECT"] = False  # Lazy connection
    app.config["MONGO_MAXPOOLSIZE"] = 50
    app.config["MONGO_MINPOOLSIZE"] = 5
    app.config["MONGO_MAXIDLETIMEMS"] = 30000
    app.config["MONGO_SERVERSELECTIONTIMEOUTMS"] = 5000

    if mongo is not None:
        try:
            mongo.init_app(app)
            logger.info("MongoDB connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {e}")
            return False

    # Test database connection and create indexes
    if mongo is not None:
        try:
            # Test connection
            mongo.db.command('ping')
            logger.info("MongoDB connection test successful")

            # Ensure required collections and indexes exist
            users = mongo.db.users  # type: ignore[attr-defined]

            # Create unique indexes for fast lookups and to enforce uniqueness
            try:
                users.create_index("email", unique=True, background=True)
                users.create_index("username", unique=True, background=True)
                users.create_index("phone", unique=True, sparse=True, background=True)
                logger.info("Database indexes created successfully")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")

        except Exception as e:
            # Handle both connection errors and other database errors
            if "ServerSelectionTimeoutError" in str(type(e)) or "ConnectionFailure" in str(type(e)):
                logger.error(f"MongoDB connection failed: {e}")
                logger.warning("Application will continue without database functionality")
            else:
                logger.error(f"Unexpected database error: {e}")

    # Try to import backend modules if present (optional during development)
    for module_name in ("Login", "signUp", "reports", "sandbox"):
        try:
            module = import_module(f"server.{module_name}")
            # Initialize login module with mongo instance if available
            if module_name == "Login" and hasattr(module, 'init_login'):
                module.init_login(mongo)
            logger.info(f"Successfully imported module: {module_name}")
        except ImportError as e:
            logger.warning(f"Module {module_name} not found: {e}")
        except Exception as e:
            logger.error(f"Error importing module {module_name}: {e}")

    logger.info("✅ Backend initialization completed")
    return True
