from flask import Blueprint, request, jsonify, session, redirect, url_for, flash
from flask_pymongo import PyMongo
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timedelta
import re
import logging
from bson import ObjectId

# Create login blueprint
login_bp = Blueprint('login', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_login(mongo_instance):
    """Initialize login module with MongoDB instance"""
    global mongo
    mongo = mongo_instance

def _get_mongo():
    """Return a usable Mongo instance, falling back to server.db.mongo if needed."""
    try:
        if mongo:
            return mongo
    except NameError:
        pass
    try:
        from server.db import mongo as db_mongo  # local import to avoid cycles
        return db_mongo
    except Exception:
        return None

def authenticate_user(email, password):
    """Authenticate user credentials"""
    try:
        mongo_inst = _get_mongo()
        if not mongo_inst:
            return None, "Database connection not available"
        
        # Find user by email
        user = mongo_inst.db.users.find_one({"email": email.lower()})
        
        if not user:
            return None, "Invalid email or password"
        
        # Require verified accounts
        if not user.get('is_verified', False):
            return None, "Account not verified. Please complete email verification."

        # Check if account is active
        if not user.get('active', True):
            return None, "Account is deactivated. Please contact support."
        
        # Check if account is locked
        if user.get('locked', False):
            lockout_time = user.get('lockout_time')
            if lockout_time and datetime.now() < lockout_time:
                remaining_time = lockout_time - datetime.now()
                return None, f"Account is locked. Try again in {int(remaining_time.total_seconds() / 60)} minutes."
        
        # Verify password
        password_hash_value = user.get('password_hash') or user.get('password')
        if not password_hash_value or not check_password_hash(password_hash_value, password):
            # Increment failed login attempts
            failed_attempts = user.get('failed_login_attempts', 0) + 1
            mongo_inst.db.users.update_one(
                {"_id": user['_id']},
                {"$set": {"failed_login_attempts": failed_attempts}}
            )
            
            # Lock account after 5 failed attempts
            if failed_attempts >= 5:
                lockout_time = datetime.now() + timedelta(minutes=30)
                mongo_inst.db.users.update_one(
                    {"_id": user['_id']},
                    {"$set": {"locked": True, "lockout_time": lockout_time}}
                )
                return None, "Account locked due to multiple failed login attempts. Try again in 30 minutes."
            
            return None, "Invalid email or password"
        
        # Reset failed login attempts on successful login
        mongo_inst.db.users.update_one(
            {"_id": user['_id']},
            {"$set": {"failed_login_attempts": 0, "locked": False, "lockout_time": None}}
        )
        
        # Update last login time
        from datetime import timezone
        mongo_inst.db.users.update_one(
            {"_id": user['_id']},
            {"$set": {"last_login": datetime.now(timezone.utc)}}
        )
        
        # Remove password from user object before returning
        user.pop('password', None)
        user.pop('password_hash', None)
        return user, "Login successful"
        
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return None, "An error occurred during authentication"

def create_user_session(user):
    """Create user session"""
    session['user_id'] = str(user['_id'])
    session['email'] = user['email']
    session['username'] = user.get('username', '')
    session['role'] = user.get('role', 'user')
    session['authenticated'] = True
    session.permanent = True

def clear_user_session():
    """Clear user session"""
    session.pop('user_id', None)
    session.pop('email', None)
    session.pop('username', None)
    session.pop('role', None)
    session.pop('authenticated', None)

def is_authenticated():
    """Check if user is authenticated"""
    return session.get('authenticated', False)

def get_current_user():
    """Get current authenticated user"""
    if not is_authenticated():
        return None
    
    try:
        user_id = session.get('user_id')
        mongo_inst = _get_mongo()
        if not user_id or not mongo_inst:
            return None
        
        try:
            obj_id = ObjectId(user_id)
        except Exception:
            return None
        user = mongo_inst.db.users.find_one({"_id": obj_id}, {"password": 0, "password_hash": 0})
        return user
    except Exception as e:
        logger.error(f"Error getting current user: {str(e)}")
        return None

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def require_auth(f):
    """Decorator to require authentication"""
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            if request.is_json:
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for('login.login_page'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def require_role(required_role):
    """Decorator to require specific role"""
    def decorator(f):
        def decorated_function(*args, **kwargs):
            if not is_authenticated():
                if request.is_json:
                    return jsonify({"error": "Authentication required"}), 401
                return redirect(url_for('login.login_page'))
            
            user = get_current_user()
            if not user or user.get('role', 'user') != required_role:
                if request.is_json:
                    return jsonify({"error": "Insufficient permissions"}), 403
                flash("You don't have permission to access this page", "error")
                return redirect(url_for('dashboard'))
            
            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__
        return decorated_function
    return decorator

@login_bp.route('/login', methods=['GET'])
def login_page():
    """Display login page"""
    if is_authenticated():
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@login_bp.route('/api/login', methods=['POST'])
def login_api():
    """Handle login API request"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Validate input
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        if not validate_email(email):
            return jsonify({"error": "Invalid email format"}), 400
        
        # Authenticate user
        user, message = authenticate_user(email, password)
        
        if not user:
            return jsonify({"error": message}), 401
        
        # Create session
        create_user_session(user)
        
        logger.info(f"User {email} logged in successfully")
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": str(user['_id']),
                "email": user['email'],
                "username": user.get('username', ''),
                "role": user.get('role', 'user')
            }
        })
        
    except Exception as e:
        logger.error(f"Login API error: {str(e)}")
        return jsonify({"error": "An error occurred during login"}), 500

@login_bp.route('/login', methods=['POST'])
def login_form():
    """Handle login form submissions from Login.html"""
    try:
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash("Email and password are required", "error")
            return redirect(url_for('login'))

        if not validate_email(email):
            flash("Invalid email format", "error")
            return redirect(url_for('login'))

        user, message = authenticate_user(email, password)
        if not user:
            logger.info(f"Login failed for {email}: {message}")
            flash(message, "error")
            return redirect(url_for('login'))

        create_user_session(user)
        flash("Logged in successfully", "success")
        return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"Login form error: {str(e)}")
        flash("An error occurred during login", "error")
        return redirect(url_for('login'))

@login_bp.route('/logout', methods=['POST'])
def logout_api():
    """Handle logout API request"""
    try:
        if is_authenticated():
            logger.info(f"User {session.get('email')} logged out")
        
        clear_user_session()
        
        return jsonify({
            "success": True,
            "message": "Logged out successfully"
        })
        
    except Exception as e:
        logger.error(f"Logout API error: {str(e)}")
        return jsonify({"error": "An error occurred during logout"}), 500

@login_bp.route('/logout')
def logout():
    """Handle logout request"""
    clear_user_session()
    flash("You have been logged out successfully", "info")
    return redirect(url_for('login'))

@login_bp.route('/app/templates/settings', methods=['GET'])
@require_auth
def get_user_profile():
    """Get current user profile"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        return jsonify({
            "success": True,
            "user": {
                "id": str(user['_id']),
                "email": user['email'],
                "username": user.get('username', ''),
                "role": user.get('role', 'user'),
                "created_at": user.get('created_at'),
                "last_login": user.get('last_login')
            }
        })
        
    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        return jsonify({"error": "An error occurred while fetching profile"}), 500

@login_bp.route('/api/user/change-password', methods=['POST'])
@require_auth
def change_password():
    """Change user password"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        if not current_password or not new_password:
            return jsonify({"error": "Current password and new password are required"}), 400
        
        # Get current user
        user = get_current_user()
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Verify current password
        mongo_inst = _get_mongo()
        if not mongo_inst:
            return jsonify({"error": "Database connection not available"}), 500
        user_with_password = mongo_inst.db.users.find_one({"_id": user['_id']})
        if not check_password_hash(user_with_password['password'], current_password):
            return jsonify({"error": "Current password is incorrect"}), 400
        
        # Validate new password
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Update password
        hashed_password = generate_password_hash(new_password)
        mongo_inst.db.users.update_one(
            {"_id": user['_id']},
            {"$set": {"password": hashed_password, "updated_at": datetime.now()}}
        )
        
        logger.info(f"User {user['email']} changed password")
        
        return jsonify({
            "success": True,
            "message": "Password changed successfully"
        })
        
    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        return jsonify({"error": "An error occurred while changing password"}), 500

@login_bp.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Check authentication status"""
    try:
        if is_authenticated():
            user = get_current_user()
            return jsonify({
                "authenticated": True,
                "user": {
                    "id": str(user['_id']),
                    "email": user['email'],
                    "username": user.get('username', ''),
                    "role": user.get('role', 'user')
                }
            })
        else:
            return jsonify({"authenticated": False})
            
    except Exception as e:
        logger.error(f"Auth status error: {str(e)}")
        return jsonify({"authenticated": False})

# Initialize mongo variable
mongo = None
