from flask import Flask

# Initialize the Flask app (paths are relative to this package directory)
app = Flask(__name__, template_folder='templates', static_folder='static')

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

# Debug: print all registered routes
try:
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
except Exception:
    pass
