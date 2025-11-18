"""
URL CLEANING - SINGLE SOURCE OF TRUTH
This function is imported by BOTH train_model.py and app.py
"""

import re
from urllib.parse import urlparse


def clean_url(url):
    """
    Clean URL - EXACT SAME STEPS EVERY TIME
    
    Steps (IN THIS ORDER):
    1. Check if valid
    2. Strip whitespace
    3. Check if starts with https (BEFORE removing it)
    4. Remove http:// or https://
    5. Remove www.
    6. Remove trailing /
    7. Lowercase everything
    8. Add back http:// or https:// based on step 3
    """
    # Step 1: Invalid check
    if not url or str(url).strip() == '' or str(url).lower() == 'nan':
        return None
    
    # Step 2: Strip whitespace
    url = str(url).strip()
    
    # Step 3: Remember if it was https BEFORE we remove it
    was_https = url.lower().startswith('https://')
    
    # Step 4: Remove protocol
    url = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
    
    # Step 5: Remove www.
    url = re.sub(r'^www\.', '', url, flags=re.IGNORECASE)
    
    # Step 6: Remove trailing slash
    url = url.rstrip('/')
    
    # Step 7: Lowercase
    url = url.lower()
    
    # Step 8: Add protocol back (use original protocol)
    if was_https:
        url = 'https://' + url
    else:
        url = 'http://' + url
    
    return url


def extract_url_features(url):
    """
    Extract 25 features from URL
    SAME FUNCTION USED IN TRAINING AND INFERENCE
    """
    features = {}
    
    # Ensure has protocol
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Character counts
    features['url_length'] = len(url)
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
    
    # IP address
    ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    features['has_ip'] = 1 if ip_pattern.search(url) else 0
    
    # Subdomain count
    try:
        parsed = urlparse(url)
        domain_parts = parsed.netloc.split('.')
        features['subdomain_count'] = max(0, len(domain_parts) - 2)
    except:
        features['subdomain_count'] = 0
    
    # HTTPS
    features['is_https'] = 1 if url.startswith('https://') else 0
    
    # Domain and path length
    try:
        parsed = urlparse(url)
        features['domain_length'] = len(parsed.netloc)
        features['path_length'] = len(parsed.path)
    except:
        features['domain_length'] = 0
        features['path_length'] = 0
    
    # Digit and letter counts
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_letters'] = sum(c.isalpha() for c in url)
    
    # Return in consistent order
    return [
        features['url_length'], features['num_dots'], features['num_hyphens'],
        features['num_underscores'], features['num_slashes'], features['num_question'],
        features['num_equal'], features['num_at'], features['num_ampersand'],
        features['num_exclamation'], features['num_space'], features['num_tilde'],
        features['num_comma'], features['num_plus'], features['num_asterisk'],
        features['num_hashtag'], features['num_dollar'], features['num_percent'],
        features['has_ip'], features['subdomain_count'], features['is_https'],
        features['domain_length'], features['path_length'], features['num_digits'],
        features['num_letters']
    ]