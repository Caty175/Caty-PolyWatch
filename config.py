"""
Configuration management for PolyWatch application.
Handles environment-specific settings and secure credential management.
"""
import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class with common settings."""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-change-me-in-production')
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Database settings
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/polywatch')
    MONGO_CONNECT = False  # Lazy connection
    MONGO_MAXPOOLSIZE = int(os.getenv('MONGO_MAXPOOLSIZE', '50'))
    MONGO_MINPOOLSIZE = int(os.getenv('MONGO_MINPOOLSIZE', '5'))
    MONGO_MAXIDLETIMEMS = int(os.getenv('MONGO_MAXIDLETIMEMS', '30000'))
    MONGO_SERVERSELECTIONTIMEOUTMS = int(os.getenv('MONGO_SERVERSELECTIONTIMEOUTMS', '5000'))
    
    # Email settings
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_EMAIL = os.getenv('SMTP_EMAIL')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
    
    # Security settings
    PASSWORD_MIN_LENGTH = int(os.getenv('PASSWORD_MIN_LENGTH', '12'))
    MAX_LOGIN_ATTEMPTS = int(os.getenv('MAX_LOGIN_ATTEMPTS', '5'))
    LOCKOUT_DURATION_MINUTES = int(os.getenv('LOCKOUT_DURATION_MINUTES', '30'))
    OTP_EXPIRY_MINUTES = int(os.getenv('OTP_EXPIRY_MINUTES', '5'))
    
    # ML Model settings
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'ML model/models')
    RF_MODEL_FILE = os.getenv('RF_MODEL_FILE', 'rf_model.joblib')
    LSTM_MODEL_FILE = os.getenv('LSTM_MODEL_FILE', 'lstm_model')
    
    # API settings
    WINSEC_SERVER_URL = os.getenv('WINSEC_SERVER', 'http://127.0.0.1:8000')
    WINSEC_API_KEY = os.getenv('WINSEC_API_KEY', 'replace_with_secure_key')
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'polywatch.log')
    
    @staticmethod
    def validate_config():
        """Validate critical configuration settings."""
        errors = []
        
        if not Config.SECRET_KEY or Config.SECRET_KEY == 'dev-secret-change-me-in-production':
            errors.append("SECRET_KEY must be set to a secure value in production")
        
        if not Config.SMTP_EMAIL or not Config.SMTP_PASSWORD:
            errors.append("SMTP_EMAIL and SMTP_PASSWORD must be set for email functionality")
        
        if not os.path.exists(Config.ML_MODEL_PATH):
            errors.append(f"ML model path does not exist: {Config.ML_MODEL_PATH}")
        
        return errors

class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    
    # More verbose logging in development
    LOG_LEVEL = 'DEBUG'
    
    # Relaxed security for development
    PASSWORD_MIN_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 10

class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    
    # Strict security in production
    PASSWORD_MIN_LENGTH = 12
    MAX_LOGIN_ATTEMPTS = 5
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    
    @staticmethod
    def validate_config():
        """Additional production-specific validation."""
        errors = Config.validate_config()
        
        if Config.SECRET_KEY == 'dev-secret-change-me-in-production':
            errors.append("SECRET_KEY must be changed from default value in production")
        
        if Config.DEBUG:
            errors.append("DEBUG must be False in production")
        
        return errors

class TestingConfig(Config):
    """Testing environment configuration."""
    DEBUG = True
    TESTING = True
    
    # Use in-memory database for testing
    MONGO_URI = 'mongodb://localhost:27017/polywatch_test'
    
    # Disable email sending in tests
    SMTP_EMAIL = None
    SMTP_PASSWORD = None
    
    # Faster testing
    OTP_EXPIRY_MINUTES = 1
    LOCKOUT_DURATION_MINUTES = 1

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env_name=None):
    """Get configuration class based on environment."""
    if env_name is None:
        env_name = os.getenv('FLASK_ENV', 'default')
    
    return config_map.get(env_name, DevelopmentConfig)
