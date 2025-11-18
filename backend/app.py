"""
Silver Umbrella - Flask Backend API
Uses SHARED url_cleaner.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from datetime import datetime
from config import Config, rate_limit_store
from explainer import explainer
from url_cleaner import clean_url, extract_url_features

app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)

print("🔄 Loading model...")
try:
    with open(Config.MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        accuracy = model_data['accuracy']
    print(f"✓ Model loaded! Accuracy: {accuracy*100:.2f}%")
except Exception as e:
    print(f"❌ Error: {e}")
    model = None
    scaler = None
    accuracy = 0


def check_rate_limit(ip):
    """Rate limit check"""
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
    return jsonify({
        'service': 'Silver Umbrella API',
        'version': '1.0.0',
        'status': 'ok' if model else 'no_model'
    })


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy' if model else 'degraded',
        'model_loaded': model is not None,
        'accuracy': f"{accuracy*100:.2f}%" if model else 'N/A'
    })


@app.route('/api/check-url', methods=['POST'])
def check_url():
    """Check URL - uses SHARED url_cleaner"""
    
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
    
    try:
        # CLEAN URL USING SHARED FUNCTION
        cleaned_url = clean_url(url)
        
        if not cleaned_url:
            return jsonify({'error': 'Invalid URL'}), 400
        
        # EXTRACT FEATURES USING SHARED FUNCTION
        features = extract_url_features(cleaned_url)
        features = np.array(features).reshape(1, -1)
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Debug
        print(f"Original: {url}")
        print(f"Cleaned: {cleaned_url}")
        print(f"Features[0:5]: {features[0][:5]}")
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        # Build feature dict
        feature_names = [
            'url_length', 'num_dots', 'num_hyphens', 'num_underscores',
            'num_slashes', 'num_question', 'num_equal', 'num_at',
            'num_ampersand', 'num_exclamation', 'num_space', 'num_tilde',
            'num_comma', 'num_plus', 'num_asterisk', 'num_hashtag',
            'num_dollar', 'num_percent', 'has_ip', 'subdomain_count',
            'is_https', 'domain_length', 'path_length', 'num_digits', 'num_letters'
        ]
        
        feature_dict = {name: int(features[0][i]) for i, name in enumerate(feature_names)}
        
        # Generate explanation
        explanation = explainer.generate_explanation(
            url=cleaned_url,
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
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Processing failed', 'message': str(e)}), 500


if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"🛡️  SILVER UMBRELLA API")
    print(f"{'='*70}\n")
    print(f"Host: {Config.HOST}:{Config.PORT}")
    print(f"Accuracy: {accuracy*100:.2f}%" if model else "No model")
    print(f"\n{'='*70}\n")
    
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)