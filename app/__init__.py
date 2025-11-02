from flask import Flask
import os
import logging
from config import get_config

# Get configuration based on environment
config_class = get_config()

# Initialize the Flask app (paths are relative to this package directory)
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(config_class)

# Configure logging
logging.basicConfig(
    level=getattr(logging, app.config['LOG_LEVEL']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(app.config['LOG_FILE']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Validate configuration
config_errors = config_class.validate_config()
if config_errors:
    for error in config_errors:
        logger.warning(f"Configuration warning: {error}")
    if app.config['FLASK_ENV'] == 'production' and config_errors:
        logger.error("Critical configuration errors in production environment")
        raise RuntimeError("Invalid production configuration")

# Import and register routes
from app import routes

# Initialize backend services (MongoDB, etc.) if available
try:
    from server.db import init_backend
    init_backend(app)
except Exception:
    # Backend is optional during initial setup; ignore if missing/misconfigured
    pass

# Register backend blueprints
try:
    from server.signUp import signup_bp
    app.register_blueprint(signup_bp)
except Exception:
    pass

try:
    from server.Login import login_bp
    app.register_blueprint(login_bp)
except Exception:
    pass

# Debug: print all registered routes
try:
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
except Exception:
    pass
