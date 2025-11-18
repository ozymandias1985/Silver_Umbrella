#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import os
import re
from urllib.parse import urlparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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

print("\n"+"="*70)
print("TRAINING MODEL")
print("="*70+"\n")

df = pd.read_csv('data/raw/raw_urls.csv')
if 'url' in df.columns and 'status' in df.columns:
    df.columns = ['url', 'label']
elif len(df.columns) == 2:
    df.columns = ['url', 'label']

print(f"Loaded {len(df):,} URLs")
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)
df = df[df['label'].isin([0, 1])]

print("Cleaning URLs...")
df['url'] = df['url'].apply(lambda x: clean_url(x))
df = df.dropna()
print(f"Cleaned {len(df):,} URLs\n")

print("Extracting features...")
features_list = []
for idx, url in enumerate(df['url']):
    if idx % 50000 == 0 and idx > 0:
        print(f"  {idx:,} / {len(df):,}")
    try:
        features_list.append(extract_features(url))
    except:
        features_list.append(np.zeros(25))

X = np.array(features_list)
y = df['label'].values

print(f"\nTraining on {len(X):,} URLs...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=10, subsample=0.9, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=42, verbose=2)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
acc = accuracy_score(y_test, test_pred)
print(f"\nAccuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, test_pred, target_names=['Phishing', 'Legitimate']))

os.makedirs('model', exist_ok=True)
with open('model/phishing_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'accuracy': acc}, f)

print(f"\nModel saved! Accuracy: {acc*100:.2f}%\n")