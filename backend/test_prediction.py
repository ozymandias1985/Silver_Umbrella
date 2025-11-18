"""Debug script"""
import sys
import os
sys.path.insert(0, 'backend')

from features import clean_url, URLFeatureExtractor
import pickle

# Load model
with open('model/phishing_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']

# Test URLs
test_urls = [
    'google.com',
    'facebook.com',
    'amazon.com',
    'paypal-verify.phishing.tk'
]

print("="*70)
print("PREDICTION TEST")
print("="*70)

for url in test_urls:
    cleaned = clean_url(url, preserve_scheme=True)
    
    extractor = URLFeatureExtractor()
    features = extractor.extract_features(cleaned).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    pred = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]
    
    result = "SAFE" if pred == 1 else "DANGER"
    confidence = probs[1] if pred == 1 else probs[0]
    
    print(f"\nURL: {url}")
    print(f"Cleaned: {cleaned}")
    print(f"Result: {result} ({confidence*100:.1f}%)")
    print(f"is_https: {int(features[0][20])}")