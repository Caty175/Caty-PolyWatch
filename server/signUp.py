from flask import Blueprint, request, redirect, url_for, render_template, session, flash
from werkzeug.security import generate_password_hash
from server.db import mongo
import random
import string
from datetime import datetime, timedelta, timezone
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import re
import secrets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """Validate password strength with enhanced security requirements"""
    if len(password) < 12:
        return False, "Password must be at least 12 characters long"

    if len(password) > 128:
        return False, "Password must be less than 128 characters long"

    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"

    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"

    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"

    # Check for common weak patterns
    weak_patterns = [
        r'(.)\1{2,}',  # Three or more consecutive identical characters
        r'(012|123|234|345|456|567|678|789|890)',  # Sequential numbers
        r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)',  # Sequential letters
    ]

    for pattern in weak_patterns:
        if re.search(pattern, password.lower()):
            return False, "Password contains weak patterns (avoid sequential or repeated characters)"

    # Check against common passwords (basic check)
    common_passwords = ['password', '123456', 'qwerty', 'admin', 'letmein', 'welcome']
    if password.lower() in common_passwords:
        return False, "Password is too common, please choose a stronger password"

    return True, "Password is valid"

def send_verification_email(to_email: str, otp: str) -> bool:
    """Send verification email using Gmail SMTP (same as test.py). Returns True if sent.
    
    Requires env vars:
      SMTP_EMAIL - Gmail address
      SMTP_PASSWORD - Gmail app password
      SMTP_SERVER - SMTP server (default: smtp.gmail.com)
      SMTP_PORT - SMTP port (default: 587)
    """
    try:
        # Get SMTP credentials from environment
        sender_email = os.getenv("SMTP_EMAIL")
        password = os.getenv("SMTP_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        
        if not sender_email or not password:
            print(f"[ERROR] Missing SMTP credentials!")
            print(f"[ERROR] SMTP_EMAIL present: {bool(sender_email)}")
            print(f"[ERROR] SMTP_PASSWORD present: {bool(password)}")
            return False
        
        print(f"[DEBUG] Sending email to: {to_email}")
        print(f"[DEBUG] Using SMTP server: {smtp_server}:{smtp_port}")
        
        # Create message (same structure as test.py)
        message = MIMEMultipart("alternative")
        message["Subject"] = "Your PolyWatch Verification Code"
        message["From"] = sender_email
        message["To"] = to_email
        
        # Create HTML content for the verification email
        html = f"""\
        <html>
          <body>
            <h2>PolyWatch Account Verification</h2>
            <p>Your verification code is:</p>
            <h1 style="color: #007bff; font-size: 24px; font-weight: bold;">{otp}</h1>
            <p>This code expires in 5 minutes.</p>
            <p>If you didn't request this verification, please ignore this email.</p>
          </body>
        </html>
        """
        
        message.attach(MIMEText(html, "html"))
        
        # Send email using the same method as test.py
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, to_email, message.as_string())
            
        print(f"✅ Verification email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        print(f"❌ Error sending verification email: {e}")
        return False

signup_bp = Blueprint('signup', __name__)


@signup_bp.route('/signup', methods=['GET'])
def signup_get():
    return render_template('signUp.html')


def validate_input_data(username, email, password, confirm_password, phone=None):
    """Validate all input data with comprehensive checks"""
    errors = []

    # Username validation
    if not username or len(username.strip()) < 3:
        errors.append("Username must be at least 3 characters long")
    elif len(username) > 50:
        errors.append("Username must be less than 50 characters")
    elif not re.match(r'^[a-zA-Z0-9_-]+$', username):
        errors.append("Username can only contain letters, numbers, hyphens, and underscores")

    # Email validation
    if not email:
        errors.append("Email is required")
    elif not validate_email(email):
        errors.append("Invalid email format")
    elif len(email) > 254:
        errors.append("Email address is too long")

    # Password validation
    if not password:
        errors.append("Password is required")
    else:
        is_valid, message = validate_password(password)
        if not is_valid:
            errors.append(message)

    # Confirm password
    if password != confirm_password:
        errors.append("Passwords do not match")

    # Phone validation (if provided)
    if phone:
        # Basic phone number validation
        phone_clean = re.sub(r'[^\d+]', '', phone)
        if len(phone_clean) < 10 or len(phone_clean) > 15:
            errors.append("Invalid phone number format")

    return errors

@signup_bp.route('/signup', methods=['POST'])
def signup_post():
    if mongo is None:
        logger.error("Database not available for signup")
        return ("Database not available", 500)

    try:
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm-password', '')
        twofa_method = request.form.get('twofa_method', 'email')
        phone = request.form.get('phone', '').strip()

        # Comprehensive input validation
        validation_errors = validate_input_data(username, email, password, confirm_password, phone)
        if validation_errors:
            logger.warning(f"Signup validation failed for {email}: {validation_errors}")
            return ("; ".join(validation_errors), 400)

        if twofa_method == 'phone' and not phone:
            return ("Phone number required for phone verification", 400)

        users = mongo.db.users  # type: ignore[attr-defined]

        # Uniqueness checks with proper error handling
        try:
            if users.find_one({'email': email}):
                logger.warning(f"Signup attempt with existing email: {email}")
                return ("Email already registered", 409)
            if users.find_one({'username': username}):
                logger.warning(f"Signup attempt with existing username: {username}")
                return ("Username already taken", 409)
            if phone and users.find_one({'phone': phone}):
                logger.warning(f"Signup attempt with existing phone: {phone}")
                return ("Phone already registered", 409)
        except Exception as e:
            logger.error(f"Database error during uniqueness check: {e}")
            return ("Database error occurred", 500)

        # Generate secure password hash
        password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

        # Generate a secure OTP and expiry (5 minutes)
        otp = ''.join(secrets.choice(string.digits) for _ in range(6))
        otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)

        # Create user record with additional security fields
        user_data = {
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'phone': phone if phone else None,
            'is_verified': False,
            'twofa_method': twofa_method,
            'otp': otp,
            'otp_expires_at': otp_expires_at,
            'created_at': datetime.now(timezone.utc),
            'failed_login_attempts': 0,
            'locked': False,
            'active': True
        }

        try:
            users.insert_one(user_data)
            logger.info(f"New user created: {username} ({email})")
        except Exception as e:
            logger.error(f"Database error during user creation: {e}")
            return ("Failed to create user account", 500)

        # Send OTP via email or SMS
        if twofa_method == 'email':
            sent = send_verification_email(email, otp)
            if not sent:
                logger.error(f"Email sending failed for {email}. OTP: {otp}")
                logger.error("Please check your SMTP configuration.")
            else:
                logger.info(f"Verification email sent to {email}")
        else:
            logger.info(f"SMS OTP should be sent to phone {phone}: {otp}")

        return redirect(url_for('signup.verify_get', username=username))

    except Exception as e:
        logger.error(f"Unexpected error during signup: {e}")
        return ("An unexpected error occurred during signup", 500)


@signup_bp.route('/verify', methods=['GET'])
def verify_get():
    username = request.args.get('username', '')
    return render_template('verify_signup.html', username=username)


@signup_bp.route('/verify', methods=['POST'])
def verify_post():
    if mongo is None:
        return ("Database not available", 500)

    username = request.form.get('username', '').strip()
    code = request.form.get('code', '').strip()

    if not username or not code:
        return ("Username and code are required", 400)

    users = mongo.db.users  # type: ignore[attr-defined]
    user = users.find_one({'username': username})
    if not user:
        return ("User not found", 404)

    if user.get('is_verified'):
        return redirect(url_for('login'))

    if user.get('otp') != code:
        return ("Invalid code", 400)

    expires_at = user.get('otp_expires_at')
    if isinstance(expires_at, datetime) and expires_at < datetime.now(timezone.utc):
        return ("Code expired", 400)

    users.update_one({'_id': user['_id']}, {
        '$set': {'is_verified': True},
        '$unset': {'otp': "", 'otp_expires_at': ""}
    })

    return redirect(url_for('login'))

