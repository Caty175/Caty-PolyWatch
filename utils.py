"""
Utility functions for PolyWatch application.
Common functions used across multiple modules.
"""
import re
import secrets
import string
import hashlib
import logging
from datetime import datetime, timezone
from functools import wraps
from flask import request, jsonify
import time

logger = logging.getLogger(__name__)

# Rate limiting storage (in production, use Redis or database)
rate_limit_storage = {}

def validate_email(email: str) -> bool:
    """
    Validate email format using regex.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if email format is valid, False otherwise
    """
    if not email or len(email) > 254:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone (str): Phone number to validate
        
    Returns:
        bool: True if phone format is valid, False otherwise
    """
    if not phone:
        return False
    
    # Remove all non-digit characters except +
    phone_clean = re.sub(r'[^\d+]', '', phone)
    
    # Check length (10-15 digits is typical for international numbers)
    if len(phone_clean) < 10 or len(phone_clean) > 15:
        return False
    
    # Must start with + or digit
    if not phone_clean[0] in ['+'] + list(string.digits):
        return False
    
    return True

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length (int): Length of the token to generate
        
    Returns:
        str: Secure random token
    """
    return secrets.token_urlsafe(length)

def generate_otp(length: int = 6) -> str:
    """
    Generate a secure numeric OTP.
    
    Args:
        length (int): Length of the OTP
        
    Returns:
        str: Numeric OTP
    """
    return ''.join(secrets.choice(string.digits) for _ in range(length))

def hash_string(text: str, salt: str = None) -> str:
    """
    Hash a string using SHA-256 with optional salt.
    
    Args:
        text (str): Text to hash
        salt (str): Optional salt to add
        
    Returns:
        str: Hexadecimal hash
    """
    if salt:
        text = text + salt
    
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def sanitize_input(text: str, max_length: int = None) -> str:
    """
    Sanitize user input by removing potentially dangerous characters.
    
    Args:
        text (str): Input text to sanitize
        max_length (int): Maximum allowed length
        
    Returns:
        str: Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    # Limit length if specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """
    Rate limiting decorator.
    
    Args:
        max_requests (int): Maximum requests allowed in the time window
        window_seconds (int): Time window in seconds
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client identifier (IP address)
            client_id = request.environ.get('REMOTE_ADDR', 'unknown')
            current_time = time.time()
            
            # Clean old entries
            cutoff_time = current_time - window_seconds
            if client_id in rate_limit_storage:
                rate_limit_storage[client_id] = [
                    timestamp for timestamp in rate_limit_storage[client_id]
                    if timestamp > cutoff_time
                ]
            
            # Check rate limit
            if client_id not in rate_limit_storage:
                rate_limit_storage[client_id] = []
            
            if len(rate_limit_storage[client_id]) >= max_requests:
                logger.warning(f"Rate limit exceeded for client {client_id}")
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            # Add current request
            rate_limit_storage[client_id].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def get_client_ip() -> str:
    """
    Get the client's IP address, handling proxies.
    
    Returns:
        str: Client IP address
    """
    # Check for forwarded IP (behind proxy)
    if 'X-Forwarded-For' in request.headers:
        return request.headers['X-Forwarded-For'].split(',')[0].strip()
    elif 'X-Real-IP' in request.headers:
        return request.headers['X-Real-IP']
    else:
        return request.environ.get('REMOTE_ADDR', 'unknown')

def log_security_event(event_type: str, details: dict, user_id: str = None):
    """
    Log security-related events for monitoring and auditing.
    
    Args:
        event_type (str): Type of security event
        details (dict): Event details
        user_id (str): Optional user ID associated with the event
    """
    log_entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'event_type': event_type,
        'client_ip': get_client_ip(),
        'user_agent': request.headers.get('User-Agent', 'unknown'),
        'user_id': user_id,
        'details': details
    }
    
    logger.warning(f"SECURITY_EVENT: {log_entry}")

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """
    Format datetime object to string.
    
    Args:
        dt (datetime): Datetime object to format
        format_str (str): Format string
        
    Returns:
        str: Formatted datetime string
    """
    if not dt:
        return "N/A"
    
    # Ensure timezone-aware datetime
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.strftime(format_str)

def safe_int(value, default: int = 0) -> int:
    """
    Safely convert value to integer.
    
    Args:
        value: Value to convert
        default (int): Default value if conversion fails
        
    Returns:
        int: Converted integer or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        value: Value to convert
        default (float): Default value if conversion fails
        
    Returns:
        float: Converted float or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
