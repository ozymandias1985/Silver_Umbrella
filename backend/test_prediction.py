"""
Debug script to see what's happening
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from features import clean_url, URLFeatureExtractor
import pickle
import numpy as np

# Load model
with open('model/phishing_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']

# Test URLs
test_urls = [
    'google.com',
    'https://google.com',
    'facebook.com',
    'amazon.com',
    'paypal-secure-verify.phishing.tk',
    'secure-login-account.suspicious.com'
]

print("="*70)
print("PREDICTION TEST")
print("="*70)

for url in test_urls:
    # Clean URL
    cleaned = clean_url(url, preserve_scheme=True)
    
    # Extract features
    extractor = URLFeatureExtractor()
    features = extractor.extract_features(cleaned).reshape(1, -1)
    
    # Scale
    features_scaled = scaler.transform(features)
    
    # Predict
    pred = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]
    
    result = "SAFE" if pred == 1 else "DANGER"
    confidence = probs[1] if pred == 1 else probs[0]
    
    print(f"\nURL: {url}")
    print(f"Cleaned: {cleaned}")
    print(f"Prediction: {result} (confidence: {confidence*100:.1f}%)")
    print(f"is_https feature: {features[0][20]}")
    print(f"url_length: {features[0][0]}")
    print(f"num_hyphens: {features[0][2]}")