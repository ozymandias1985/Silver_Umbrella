"""
Silver Umbrella - Shared Feature Extraction
SINGLE SOURCE OF TRUTH for URL cleaning and feature extraction
"""

import re
import numpy as np
from urllib.parse import urlparse


def clean_url(url, preserve_scheme=True):
    """
    Clean and normalize URL - EXACT SAME LOGIC FOR TRAINING AND INFERENCE
    
    This function MUST be used by both train_model.py and app.py
    Any changes here automatically apply to both training and inference
    
    Args:
        url: Raw URL string
        preserve_scheme: If True, keeps https:// vs http:// (CRITICAL for is_https feature)
    
    Returns:
        Cleaned URL string or None if invalid
    """
    # Step 1: Handle invalid input
    if not url or str(url).strip() == '' or str(url).lower() == 'nan':
        return None
    
    # Step 2: Convert to string and strip whitespace
    url = str(url).strip()
    
    # Step 3: Check if original URL used HTTPS (before we modify it)
    original_is_https = url.lower().startswith('https://')
    
    # Step 4: Remove any existing protocol (http:// or https://)
    url = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
    
    # Step 5: Remove www. prefix
    url = re.sub(r'^www\.', '', url, flags=re.IGNORECASE)
    
    # Step 6: Remove trailing slashes
    url = url.rstrip('/')
    
    # Step 7: Convert to lowercase
    url = url.lower()
    
    # Step 8: Final whitespace cleanup
    url = url.strip()
    
    # Step 9: Add protocol back (preserve HTTPS if original had it)
    if preserve_scheme and original_is_https:
        url = 'https://' + url
    else:
        url = 'http://' + url
    
    return url


class URLFeatureExtractor:
    """
    Extract 25 numerical features from URLs
    EXACT SAME LOGIC FOR TRAINING AND INFERENCE
    """
    
    @staticmethod
    def extract_features(url):
        """
        Extract 25 features from a URL
        
        CRITICAL: This function MUST produce identical output for training and inference
        
        Args:
            url: Cleaned URL string (already processed by clean_url function!)
        
        Returns:
            numpy array of exactly 25 features in this order:
            [url_length, num_dots, num_hyphens, num_underscores, num_slashes,
             num_question, num_equal, num_at, num_ampersand, num_exclamation,
             num_space, num_tilde, num_comma, num_plus, num_asterisk,
             num_hashtag, num_dollar, num_percent, has_ip, subdomain_count,
             is_https, domain_length, path_length, num_digits, num_letters]
        """
        features = {}
        
        # Ensure URL has protocol (shouldn't be needed if clean_url was used, but safety check)
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Feature 1: Total URL length
        features['url_length'] = len(url)
        
        # Features 2-18: Character counts (common phishing indicators)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question'] = url.count('?')
        features['num_equal'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_ampersand'] = url.count('&')
        features['num_exclamation'] = url.count('!')
        features['num_space'] = url.count('%20')
        features['num_tilde'] = url.count('~')
        features['num_comma'] = url.count(',')
        features['num_plus'] = url.count('+')
        features['num_asterisk'] = url.count('*')
        features['num_hashtag'] = url.count('#')
        features['num_dollar'] = url.count('$')
        features['num_percent'] = url.count('%')
        
        # Feature 19: IP address in URL (major phishing indicator)
        ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        features['has_ip'] = 1 if ip_pattern.search(url) else 0
        
        # Feature 20: Number of subdomains
        try:
            parsed = urlparse(url)
            domain_parts = parsed.netloc.split('.')
            # Normal domain has 2 parts (example.com), subdomains add more
            features['subdomain_count'] = max(0, len(domain_parts) - 2)
        except:
            features['subdomain_count'] = 0
        
        # Feature 21: HTTPS usage (CRITICAL - must match training!)
        features['is_https'] = 1 if url.startswith('https://') else 0
        
        # Feature 22: Domain length
        try:
            parsed = urlparse(url)
            features['domain_length'] = len(parsed.netloc)
        except:
            features['domain_length'] = 0
        
        # Feature 23: Path length
        try:
            parsed = urlparse(url)
            features['path_length'] = len(parsed.path)
        except:
            features['path_length'] = 0
        
        # Feature 24: Number of digits
        features['num_digits'] = sum(c.isdigit() for c in url)
        
        # Feature 25: Number of letters
        features['num_letters'] = sum(c.isalpha() for c in url)
        
        # CRITICAL: Return features in EXACT ORDER as numpy array
        feature_order = [
            'url_length', 'num_dots', 'num_hyphens', 'num_underscores',
            'num_slashes', 'num_question', 'num_equal', 'num_at',
            'num_ampersand', 'num_exclamation', 'num_space', 'num_tilde',
            'num_comma', 'num_plus', 'num_asterisk', 'num_hashtag',
            'num_dollar', 'num_percent', 'has_ip', 'subdomain_count',
            'is_https', 'domain_length', 'path_length', 'num_digits', 'num_letters'
        ]
        
        return np.array([features[name] for name in feature_order])