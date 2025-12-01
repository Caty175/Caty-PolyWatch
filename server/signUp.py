from flask import Blueprint, request, redirect, url_for, render_template
from werkzeug.security import generate_password_hash
from server.db import mongo
import random
import string
from datetime import datetime, timedelta
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
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


@signup_bp.route('/signup', methods=['POST'])
def signup_post():
    if mongo is None:
        # Backend not initialized with Mongo; fail fast with clear message
        return ("Database not available", 500)

    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    confirm_password = request.form.get('confirm-password', '')

    twofa_method = request.form.get('twofa_method', 'email')
    phone = request.form.get('phone', '').strip()

    if not username or not email or not password or not confirm_password:
        return ("All fields are required", 400)
    if password != confirm_password:
        return ("Passwords do not match", 400)
    if twofa_method == 'phone' and not phone:
        return ("Phone number required for phone verification", 400)

    users = mongo.db.users  # type: ignore[attr-defined]

    # Uniqueness checks
    if users.find_one({'email': email}):
        return ("Email already registered", 409)
    if users.find_one({'username': username}):
        return ("Username already taken", 409)
    if phone and users.find_one({'phone': phone}):
        return ("Phone already registered", 409)

    password_hash = generate_password_hash(password)

    # Generate an OTP and expiry (5 minutes)
    otp = ''.join(random.choices(string.digits, k=6))
    otp_expires_at = datetime.utcnow() + timedelta(minutes=5)

    users.insert_one({
        'username': username,
        'email': email,
        'password_hash': password_hash,
        'phone': phone if phone else None,
        'is_verified': False,
        'twofa_method': twofa_method,
        'otp': otp,
        'otp_expires_at': otp_expires_at,
    })

    # Send OTP via email or SMS
    if twofa_method == 'email':
        sent = send_verification_email(email, otp)
        if not sent:
            print(f"[2FA] Email sending failed. OTP for {email}: {otp}")
            print(f"[2FA] Please check your SMTP configuration.")
        else:
            print(f"[2FA] Verification email sent to {email}")
    else:
        print(f"[2FA] Send OTP to phone {phone}: {otp}")

    return redirect(url_for('signup.verify_get', username=username))


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
    if isinstance(expires_at, datetime) and expires_at < datetime.utcnow():
        return ("Code expired", 400)

    users.update_one({'_id': user['_id']}, {
        '$set': {'is_verified': True},
        '$unset': {'otp': "", 'otp_expires_at': ""}
    })

    return redirect(url_for('login'))

