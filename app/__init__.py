from flask import Flask

# Initialize the Flask app (paths are relative to this package directory)
app = Flask(__name__, template_folder='templates', static_folder='static')

# Import and register routes
from app import routes
