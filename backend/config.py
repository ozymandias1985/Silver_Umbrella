"""
Silver Umbrella Backend Configuration
"""

import os
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

class Config:
    """Configuration settings for Silver Umbrella backend"""
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = 'gpt-3.5-turbo'
    
    # Model paths
    MODEL_PATH = 'model\phishing_model.pkl'
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_PER_IP = 50
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.70
    LOW_CONFIDENCE_THRESHOLD = 0.70
    
    # Flask settings
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # CORS settings
    CORS_ORIGINS = '*'
    
    # Feature names
    FEATURE_NAMES = [
        'url_length', 'num_dots', 'num_hyphens', 'num_underscores',
        'num_slashes', 'num_question', 'num_equal', 'num_at',
        'num_ampersand', 'num_exclamation', 'num_space', 'num_tilde',
        'num_comma', 'num_plus', 'num_asterisk', 'num_hashtag',
        'num_dollar', 'num_percent', 'has_ip', 'subdomain_count',
        'is_https', 'domain_length', 'path_length', 'num_digits', 'num_letters'
    ]

# Rate limiting storage
rate_limit_store = {}