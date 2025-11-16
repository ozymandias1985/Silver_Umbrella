#!/usr/bin/env python3
"""
Silver Umbrella - Data Cleaning & SVM Model Training
Trains on URLs to detect phishing (0) vs legitimate (1)
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
from urllib.parse import urlparse
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class URLFeatureExtractor:
    """Extract numerical features from URLs for ML"""
    
    @staticmethod
    def extract_features(url):
        """Extract 25 features from a URL"""
        features = {}
        
        # Ensure URL has protocol for parsing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Basic character counts (phishing indicators)
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
        
        # Check for IP address (major phishing indicator)
        ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        features['has_ip'] = 1 if ip_pattern.search(url) else 0
        
        # Subdomain count
        try:
            parsed = urlparse(url)
            domain_parts = parsed.netloc.split('.')
            features['subdomain_count'] = max(0, len(domain_parts) - 2)
        except:
            features['subdomain_count'] = 0
        
        # Check for HTTPS
        features['is_https'] = 1 if url.startswith('https://') else 0
        
        # Domain length
        try:
            parsed = urlparse(url)
            features['domain_length'] = len(parsed.netloc)
        except:
            features['domain_length'] = 0
        
        # Path length
        try:
            parsed = urlparse(url)
            features['path_length'] = len(parsed.path)
        except:
            features['path_length'] = 0
        
        # Count digits and letters
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        
        return np.array(list(features.values()))

def clean_url(url):
    """Clean and standardize URL format"""
    if pd.isna(url):
        return None
    
    url = str(url).strip().lower()
    
    # Remove common prefixes
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    
    # Remove trailing slashes
    url = url.rstrip('/')
    
    # Remove whitespace
    url = url.strip()
    
    # Add protocol back
    url = 'http://' + url
    
    return url

def load_and_clean_data(csv_path):
    """Load CSV and clean URLs"""
    print(f"\n{'='*70}")
    print(f"📊 LOADING DATA")
    print(f"{'='*70}\n")
    print(f"Reading from: {csv_path}")
    
    # Load CSV (no header, two columns: url,label)
    df = pd.read_csv(csv_path, names=['url', 'label'], header=None)
    
    print(f"✓ Loaded {len(df):,} URLs")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Clean URLs
    print(f"\n🧹 Cleaning URLs...")
    df['url'] = df['url'].apply(clean_url)
    
    # Remove any nulls
    original_count = len(df)
    df = df.dropna()
    removed = original_count - len(df)
    
    print(f"✓ Cleaned {len(df):,} URLs (removed {removed} invalid entries)")
    
    # Show label distribution
    print(f"\n📈 Label Distribution:")
    phishing_count = (df['label'] == 0).sum()
    legit_count = (df['label'] == 1).sum()
    print(f"  - Phishing (0):   {phishing_count:>8,} URLs ({phishing_count/len(df)*100:>5.1f}%)")
    print(f"  - Legitimate (1): {legit_count:>8,} URLs ({legit_count/len(df)*100:>5.1f}%)")
    
    return df

def extract_all_features(df):
    """Extract features from all URLs"""
    print(f"\n{'='*70}")
    print(f"⚙️  EXTRACTING FEATURES")
    print(f"{'='*70}\n")
    print(f"Processing {len(df):,} URLs...")
    print(f"(Progress updates every 50,000 URLs)\n")
    
    extractor = URLFeatureExtractor()
    features_list = []
    
    for idx, url in enumerate(df['url']):
        # Show progress
        if idx % 50000 == 0 and idx > 0:
            print(f"   ✓ {idx:>8,} / {len(df):,} URLs processed ({idx/len(df)*100:>5.1f}%)")
        
        try:
            features = extractor.extract_features(url)
            features_list.append(features)
        except Exception as e:
            # Use zero vector for failed URLs
            features_list.append(np.zeros(25))
    
    print(f"   ✓ {len(df):>8,} / {len(df):,} URLs processed (100.0%)")
    print(f"\n✓ Feature extraction complete!")
    
    return np.array(features_list)

def train_svm_model(X, y):
    """Train SVM model"""
    print(f"\n{'='*70}")
    print(f"🤖 TRAINING SVM MODEL")
    print(f"{'='*70}\n")
    
    # Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data Split:")
    print(f"  - Training: {len(X_train):>8,} URLs ({len(X_train)/len(X)*100:>5.1f}%)")
    print(f"  - Testing:  {len(X_test):>8,} URLs ({len(X_test)/len(X)*100:>5.1f}%)")
    
    # Scale features (normalize to mean=0, std=1)
    print(f"\n⚖️  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled")
    
    # Train SVM
    print(f"\n🔬 Training Support Vector Machine (SVM)...")
    print(f"   Algorithm: RBF kernel (handles non-linear patterns)")
    print(f"   This may take 10-20 minutes for large datasets...\n")
    
    model = SVC(
        kernel='rbf',       # Radial Basis Function kernel
        C=1.0,              # Regularization
        gamma='scale',      # Kernel coefficient
        probability=True,   # Enable confidence scores
        random_state=42,
        cache_size=1000,    # MB of cache
        verbose=True        # Show progress
    )
    
    model.fit(X_train_scaled, y_train)
    
    print(f"\n✓ SVM training complete!")
    print(f"  - Support vectors: {len(model.support_vectors_):,}")
    
    # Evaluate performance
    print(f"\n{'='*70}")
    print(f"📊 MODEL PERFORMANCE")
    print(f"{'='*70}\n")
    
    # Training accuracy
    y_train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy:  {train_accuracy*100:.2f}%")
    
    # Testing accuracy (most important!)
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Testing Accuracy:   {test_accuracy*100:.2f}% ← This is what matters!\n")
    
    # Detailed report
    print(f"Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Phishing (0)', 'Legitimate (1)']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion Matrix:")
    print(f"{'':20} Predicted")
    print(f"{'':20} Phishing    Legitimate")
    print(f"Actual Phishing     {cm[0][0]:>8,}    {cm[0][1]:>8,}")
    print(f"       Legitimate   {cm[1][0]:>8,}    {cm[1][1]:>8,}")
    
    return model, scaler, test_accuracy

def save_model(model, scaler, accuracy):
    """Save trained model to file"""
    print(f"\n{'='*70}")
    print(f"💾 SAVING MODEL")
    print(f"{'='*70}\n")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Package everything together
    model_data = {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'feature_names': [
            'url_length', 'num_dots', 'num_hyphens', 'num_underscores',
            'num_slashes', 'num_question', 'num_equal', 'num_at',
            'num_ampersand', 'num_exclamation', 'num_space', 'num_tilde',
            'num_comma', 'num_plus', 'num_asterisk', 'num_hashtag',
            'num_dollar', 'num_percent', 'has_ip', 'subdomain_count',
            'is_https', 'domain_length', 'path_length', 'num_digits', 'num_letters'
        ]
    }
    
    # Save to file
    model_path = 'model/phishing_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Get file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
    
    print(f"✓ Model saved successfully!")
    print(f"  - Location: {model_path}")
    print(f"  - File size: {file_size:.1f} MB")
    print(f"  - Accuracy: {accuracy*100:.2f}%")

def main():
    """Main training pipeline"""
    print(f"\n{'='*70}")
    print(f"🛡️  SILVER UMBRELLA - ML MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Training SVM (Support Vector Machine) for phishing detection")
    print(f"{'='*70}")
    
    # Step 1: Load data
    csv_path = 'data/raw/raw_urls.csv'
    
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: CSV file not found!")
        print(f"Expected location: {csv_path}")
        print(f"\nPlease make sure your CSV file is at:")
        print(f"  D:\\Silver_Umbrella_Prime\\data\\raw\\raw_urls.csv")
        return
    
    df = load_and_clean_data(csv_path)
    
    # Step 2: Extract features
    X = extract_all_features(df)
    y = df['label'].values
    
    # Step 3: Train model
    model, scaler, accuracy = train_svm_model(X, y)
    
    # Step 4: Save model
    save_model(model, scaler, accuracy)
    
    # Done!
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n🎯 Model ready with {accuracy*100:.2f}% accuracy!")
    print(f"📂 Saved at: model/phishing_model.pkl")
    print(f"\n🚀 Next step: Start the backend server")
    print(f"   cd backend")
    print(f"   python app.py")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()