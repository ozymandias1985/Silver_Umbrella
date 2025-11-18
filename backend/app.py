from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re
from urllib.parse import urlparse
from datetime import datetime
from config import Config, rate_limit_store

app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)

WHITELIST = {'google.com','facebook.com','amazon.com','apple.com','microsoft.com','netflix.com','twitter.com','x.com','instagram.com','linkedin.com','youtube.com','wikipedia.org','reddit.com','ebay.com','paypal.com','chase.com','bankofamerica.com','wellsfargo.com','gmail.com','yahoo.com','outlook.com','bing.com','walmart.com','target.com','bestbuy.com','homedepot.com','spotify.com','twitch.tv','github.com','stackoverflow.com','adobe.com','tiktok.com'}

def clean_url(url):
    if not url or str(url).strip() == '' or str(url).lower() == 'nan':
        return None
    url = str(url).strip()
    was_https = url.lower().startswith('https://')
    url = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
    url = re.sub(r'^www\.', '', url, flags=re.IGNORECASE)
    url = url.rstrip('/')
    url = url.lower()
    if was_https:
        url = 'https://' + url
    else:
        url = 'http://' + url
    return url

def extract_features(url):
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    f = {}
    f['url_length'] = len(url)
    f['num_dots'] = url.count('.')
    f['num_hyphens'] = url.count('-')
    f['num_underscores'] = url.count('_')
    f['num_slashes'] = url.count('/')
    f['num_question'] = url.count('?')
    f['num_equal'] = url.count('=')
    f['num_at'] = url.count('@')
    f['num_ampersand'] = url.count('&')
    f['num_exclamation'] = url.count('!')
    f['num_space'] = url.count('%20')
    f['num_tilde'] = url.count('~')
    f['num_comma'] = url.count(',')
    f['num_plus'] = url.count('+')
    f['num_asterisk'] = url.count('*')
    f['num_hashtag'] = url.count('#')
    f['num_dollar'] = url.count('$')
    f['num_percent'] = url.count('%')
    ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    f['has_ip'] = 1 if ip_pattern.search(url) else 0
    try:
        parsed = urlparse(url)
        domain_parts = parsed.netloc.split('.')
        f['subdomain_count'] = max(0, len(domain_parts) - 2)
    except:
        f['subdomain_count'] = 0
    f['is_https'] = 1 if url.startswith('https://') else 0
    try:
        parsed = urlparse(url)
        f['domain_length'] = len(parsed.netloc)
        f['path_length'] = len(parsed.path)
    except:
        f['domain_length'] = 0
        f['path_length'] = 0
    f['num_digits'] = sum(c.isdigit() for c in url)
    f['num_letters'] = sum(c.isalpha() for c in url)
    order = ['url_length','num_dots','num_hyphens','num_underscores','num_slashes','num_question','num_equal','num_at','num_ampersand','num_exclamation','num_space','num_tilde','num_comma','num_plus','num_asterisk','num_hashtag','num_dollar','num_percent','has_ip','subdomain_count','is_https','domain_length','path_length','num_digits','num_letters']
    return np.array([f[name] for name in order])

print("Loading model...")
try:
    with open(Config.MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        accuracy = model_data['accuracy']
    print(f"Model loaded! Accuracy: {accuracy*100:.2f}%")
except Exception as e:
    print(f"Error: {e}")
    model = None
    scaler = None
    accuracy = 0

def check_rate_limit(ip):
    if not Config.RATE_LIMIT_ENABLED:
        return True
    today = datetime.now().date()
    key = f"{ip}:{today}"
    if key not in rate_limit_store:
        rate_limit_store[key] = 0
    if rate_limit_store[key] >= Config.RATE_LIMIT_PER_IP:
        return False
    rate_limit_store[key] += 1
    return True

@app.route('/')
def home():
    return jsonify({'service': 'Silver Umbrella API', 'version': '1.0.0', 'status': 'ok' if model else 'no_model'})

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy' if model else 'degraded', 'model_loaded': model is not None, 'accuracy': f"{accuracy*100:.2f}%" if model else 'N/A'})

@app.route('/api/check-url', methods=['POST'])
def check_url():
    try:
        ip = request.remote_addr
        if not check_rate_limit(ip):
            return jsonify({'error': 'Rate limit exceeded'}), 429
        if not model:
            return jsonify({'error': 'Model not loaded'}), 503
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing URL'}), 400
        url = data['url'].strip()
        if not url:
            return jsonify({'error': 'Empty URL'}), 400
        
        cleaned_url = clean_url(url)
        if not cleaned_url:
            return jsonify({'error': 'Invalid URL'}), 400
        
        domain = cleaned_url.replace('http://','').replace('https://','').split('/')[0]
        if domain in WHITELIST:
            return jsonify({'url': url, 'cleaned_url': cleaned_url, 'prediction': 'legitimate', 'prediction_code': 1, 'confidence': 0.99, 'confidence_percent': 99, 'verdict': '✅ SAFE TO VISIT', 'emoji': '✅', 'color': 'green', 'explanation': 'This is a well-known, trusted website used safely by millions worldwide.', 'details': ['Recognized legitimate domain', 'Trusted worldwide'], 'advice': 'This link is safe to click.', 'timestamp': datetime.now().isoformat()}), 200
        
        features = extract_features(cleaned_url).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        if prediction == 1:
            verdict = '✅ SAFE TO VISIT'
            emoji = '✅'
            explanation = 'Our AI analysis indicates this link appears to be legitimate and safe to visit.'
            details = ['URL structure looks normal', 'No obvious phishing indicators']
        else:
            verdict = '🚨 DANGER - LIKELY PHISHING'
            emoji = '🚨'
            explanation = '⚠️ STOP! This link is dangerous and trying to steal your information. Delete it immediately and do not click it.'
            details = ['Suspicious URL patterns detected', 'Potential phishing attempt']
        
        return jsonify({'url': url, 'cleaned_url': cleaned_url, 'prediction': 'phishing' if prediction == 0 else 'legitimate', 'prediction_code': int(prediction), 'confidence': float(confidence), 'confidence_percent': int(confidence * 100), 'verdict': verdict, 'emoji': emoji, 'color': 'green' if prediction == 1 else 'red', 'explanation': explanation, 'details': details, 'advice': 'This link is safe to click.' if prediction == 1 else 'Do NOT click this link!', 'timestamp': datetime.now().isoformat()}), 200
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

if __name__ == '__main__':
    print(f"\nSILVER UMBRELLA API\nHost: {Config.HOST}:{Config.PORT}\nModel: {'Loaded' if model else 'NOT LOADED'}\n")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)