#!/usr/bin/env python3
"""
Silver Umbrella - Model Training
Uses SHARED url_cleaner.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys

# Import shared URL cleaner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from url_cleaner import clean_url, extract_url_features

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(csv_path):
    """Load and clean data"""
    print(f"\n{'='*70}")
    print(f"📊 LOADING DATA")
    print(f"{'='*70}\n")
    print(f"Reading from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'url' in df.columns and 'status' in df.columns:
        print(f"✓ Detected header with columns: {list(df.columns)}")
        df.columns = ['url', 'label']
    elif len(df.columns) == 2:
        df.columns = ['url', 'label']
    
    print(f"✓ Loaded {len(df):,} URLs")
    
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # CLEAN URLS USING SHARED FUNCTION
    print(f"\n🧹 Cleaning URLs using SHARED url_cleaner.clean_url()...")
    df['url'] = df['url'].apply(lambda x: clean_url(x))
    
    original_count = len(df)
    df = df.dropna()
    removed = original_count - len(df)
    
    print(f"✓ Cleaned {len(df):,} URLs (removed {removed:,} invalid)")
    
    print(f"\n📈 Label Distribution:")
    phishing = (df['label'] == 0).sum()
    legit = (df['label'] == 1).sum()
    total = len(df)
    
    print(f"  Phishing (0):   {phishing:>8,} ({phishing/total*100:>5.1f}%)")
    print(f"  Legitimate (1): {legit:>8,} ({legit/total*100:>5.1f}%)")
    
    return df


def extract_all_features(df):
    """Extract features"""
    print(f"\n{'='*70}")
    print(f"⚙️  EXTRACTING FEATURES")
    print(f"{'='*70}\n")
    print(f"Processing {len(df):,} URLs using SHARED extract_url_features()...")
    print(f"(Progress updates every 50,000 URLs)\n")
    
    features_list = []
    
    for idx, url in enumerate(df['url']):
        if idx % 50000 == 0 and idx > 0:
            print(f"   ✓ {idx:>8,} / {len(df):,} processed ({idx/len(df)*100:.1f}%)")
        
        try:
            features = extract_url_features(url)
            features_list.append(features)
        except:
            features_list.append([0] * 25)
    
    print(f"   ✓ {len(df):>8,} / {len(df):,} processed (100.0%)")
    print(f"\n✓ Feature extraction complete!")
    
    return np.array(features_list)


def train_model(X, y):
    """Train model"""
    print(f"\n{'='*70}")
    print(f"🤖 TRAINING MODEL")
    print(f"{'='*70}\n")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data Split:")
    print(f"  Training:  {len(X_train):>8,} URLs ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Testing:   {len(X_test):>8,} URLs ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\n⚖️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features normalized")
    
    print(f"\n🔬 Training Gradient Boosting Classifier...")
    print(f"   (Takes 8-10 minutes)\n")
    
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.9,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        verbose=2
    )
    
    model.fit(X_train_scaled, y_train)
    
    print(f"\n✓ Training complete!")
    
    print(f"\n{'='*70}")
    print(f"📊 PERFORMANCE")
    print(f"{'='*70}\n")
    
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy:  {train_acc*100:.2f}%")
    
    test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Testing Accuracy:   {test_acc*100:.2f}%\n")
    
    print(classification_report(y_test, test_pred, target_names=['Phishing', 'Legitimate']))
    
    cm = confusion_matrix(y_test, test_pred)
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Phishing  Legitimate")
    print(f"Actual Phishing     {cm[0][0]:>8,}  {cm[0][1]:>8,}")
    print(f"       Legitimate   {cm[1][0]:>8,}  {cm[1][1]:>8,}")
    
    return model, scaler, test_acc


def save_model(model, scaler, accuracy):
    """Save model"""
    print(f"\n{'='*70}")
    print(f"💾 SAVING MODEL")
    print(f"{'='*70}\n")
    
    os.makedirs('model', exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy
    }
    
    path = 'model/phishing_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    
    print(f"✓ Model saved!")
    print(f"  Location: {path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Accuracy: {accuracy*100:.2f}%")


def main():
    """Main training pipeline"""
    print(f"\n{'='*70}")
    print(f"🛡️  SILVER UMBRELLA - MODEL TRAINING")
    print(f"{'='*70}")
    print(f"✓ Using SHARED url_cleaner.py")
    print(f"{'='*70}")
    
    csv_path = 'data/raw/raw_urls.csv'
    
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: File not found: {csv_path}")
        return
    
    df = load_and_clean_data(csv_path)
    if df is None:
        return
    
    X = extract_all_features(df)
    y = df['label'].values
    
    model, scaler, accuracy = train_model(X, y)
    
    save_model(model, scaler, accuracy)
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS!")
    print(f"{'='*70}")
    print(f"\n🎯 Model accuracy: {accuracy*100:.2f}%")
    print(f"📂 Saved at: model/phishing_model.pkl")
    print(f"\n🚀 Next: python backend/app.py")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()