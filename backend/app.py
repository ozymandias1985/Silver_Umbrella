"""
Silver Umbrella - Flask Backend API
REST API for phishing detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re
from urllib.parse import urlparse
from datetime import datetime, timedelta
from config import Config, rate_limit_store
from explainer import explainer

app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)

# Load model on startup
print("🔄 Loading Silver Umbrella model...")
try:
    with open(Config.MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        model_accuracy = model_data['accuracy']
    print(f"✓ Model loaded successfully! Accuracy: {model_accuracy*100:.2f}%")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️  Make sure you've run: python train_model.py")
    model = None
    scaler = None
    model_accuracy = 0

class URLFeatureExtractor:
    """Extract features from URLs"""
    
    @staticmethod
    def extract_features(url):
        """Extract 25 numerical features from URL"""
        features = {}
        
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
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
        
        ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        features['has_ip'] = 1 if ip_pattern.search(url) else 0
        
        try:
            parsed = urlparse(url)
            domain_parts = parsed.netloc.split('.')
            features['subdomain_count'] = max(0, len(domain_parts) - 2)
        except:
            features['subdomain_count'] = 0
        
        features['is_https'] = 1 if url.startswith('https://') else 0
        
        try:
            parsed = urlparse(url)
            features['domain_length'] = len(parsed.netloc)
        except:
            features['domain_length'] = 0
        
        try:
            parsed = urlparse(url)
            features['path_length'] = len(parsed.path)
        except:
            features['path_length'] = 0
        
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        
        return np.array(list(features.values())).reshape(1, -1), features

def check_rate_limit(ip_address):
    """Check if IP has exceeded rate limit"""
    if not Config.RATE_LIMIT_ENABLED:
        return True
    
    today = datetime.now().date()
    key = f"{ip_address}:{today}"
    
    if key not in rate_limit_store:
        rate_limit_store[key] = 0
    
    if rate_limit_store[key] >= Config.RATE_LIMIT_PER_IP:
        return False
    
    rate_limit_store[key] += 1
    return True

@app.route('/')
def home():
    """API root endpoint"""
    return jsonify({
        'service': 'Silver Umbrella API',
        'version': '1.0.0',
        'status': 'operational' if model else 'model_not_loaded'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model else 'degraded',
        'model_loaded': model is not None,
        'model_accuracy': f"{model_accuracy*100:.2f}%" if model else 'N/A',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_type': 'Support Vector Machine (SVM)',
        'accuracy': f"{model_accuracy*100:.2f}%",
        'support_vectors': len(model.support_vectors_),
        'features_used': 25
    })

@app.route('/api/check-url', methods=['POST'])
def check_url():
    """Main endpoint: Check if URL is phishing or legitimate"""
    
    client_ip = request.remote_addr
    
    if not check_rate_limit(client_ip):
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': f'You have reached the maximum of {Config.RATE_LIMIT_PER_IP} checks per day.'
        }), 429
    
    if not model:
        return jsonify({
            'error': 'Model not available',
            'message': 'Please run: python train_model.py'
        }), 503
    
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({
            'error': 'Invalid request',
            'message': 'Please provide a URL: {"url": "example.com"}'
        }), 400
    
    url = data['url'].strip()
    
    if not url:
        return jsonify({
            'error': 'Empty URL',
            'message': 'Please enter a valid URL'
        }), 400
    
    try:
        feature_vector, feature_dict = URLFeatureExtractor.extract_features(url)
        feature_vector_scaled = scaler.transform(feature_vector)
        
        prediction = model.predict(feature_vector_scaled)[0]
        probabilities = model.predict_proba(feature_vector_scaled)[0]
        
        if prediction == 0:
            confidence = probabilities[0]
        else:
            confidence = probabilities[1]
        
        explanation = explainer.generate_explanation(
            url=url,
            prediction=prediction,
            confidence=confidence,
            features=feature_dict
        )
        
        response = {
            'url': url,
            'prediction': 'phishing' if prediction == 0 else 'legitimate',
            'prediction_code': int(prediction),
            'confidence': float(confidence),
            'confidence_percent': int(confidence * 100),
            'verdict': explanation['verdict'],
            'emoji': explanation['emoji'],
            'color': explanation['color'],
            'explanation': explanation['senior_friendly_summary'],
            'details': explanation['details'],
            'advice': explanation['advice'],
            'explanation_type': explanation['explanation_type'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error processing URL: {e}")
        return jsonify({
            'error': 'Processing error',
            'message': f'Unable to analyze URL: {str(e)}'
        }), 500

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"🛡️  SILVER UMBRELLA API SERVER")
    print(f"{'='*70}\n")
    print(f"🚀 Starting Flask server...")
    print(f"📡 API will be available at: http://{Config.HOST}:{Config.PORT}")
    print(f"🔒 Rate limit: {Config.RATE_LIMIT_PER_IP} checks per IP per day")
    print(f"🤖 Model accuracy: {model_accuracy*100:.2f}%" if model else "⚠️  Model not loaded!")
    print(f"\n{'='*70}\n")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )