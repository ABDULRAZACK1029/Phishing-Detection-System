
# -*- coding: utf-8 -*-
"""
Final Random Forest Trainer
Trains a robust Random Forest model with 50,000+ samples and enhanced features.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ml.feature_extraction import FeatureExtractor

warnings.filterwarnings('ignore')

class FinalRandomForestTrainer:
    """
    Trainer specifically for the enhanced Random Forest model.
    Focuses on high-volume training with advanced features.
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.feature_names = self.feature_extractor.get_feature_names()
        self.model = None
        
    def create_enhanced_synthetic_data(self, n_samples: int = 50000):
        """
        Generate a large, high-quality synthetic dataset.
        Simulates realistic patterns for all 27+ features.
        """
        print(f"[*] Generating {n_samples} enhanced synthetic samples...")
        
        X = []
        y = []
        
        np.random.seed(42)
        
        for i in range(n_samples):
            # 3 Categories: 
            # 0-40%: Easy Legitimate (simple)
            # 40-50%: Hard Legitimate (complex, academic, login pages)
            # 50-100%: Phishing
            
            rand_val = np.random.random()
            
            if rand_val >= 0.5:
                # --- PHISHING PATTERNS ---
                # High entropy domains (random strings), deep paths, IPs, suspicious TLDs
                
                # Core length features
                url_len = np.random.randint(45, 120)       # Mid-to-long
                dom_len = np.random.randint(15, 40)
                path_len = url_len - dom_len - 8
                
                # Complexity
                dots = np.random.randint(2, 6)             # sub.sub.domain.com
                hyphens = np.random.randint(1, 5)          # secure-login-verify
                slashes = np.random.randint(3, 8)          # Deep paths
                
                # Suspicious indicators
                has_ip = np.random.choice([0, 1], p=[0.7, 0.3])
                suspicious_tld = np.random.choice([0, 1], p=[0.6, 0.4])
                shortener = np.random.choice([0, 1], p=[0.8, 0.2])
                
                # New advanced features simulation
                entropy = np.random.uniform(3.5, 5.5)      # High entropy (random chars)
                digit_ratio = np.random.uniform(0.1, 0.3)  # Lots of numbers
                longest_token = np.random.randint(10, 25)  # Long random tokens
                tld_in_path = np.random.choice([0, 1], p=[0.7, 0.3]) # com-login
                client_server = np.random.choice([0, 1], p=[0.7, 0.3])
                
                label = 1
                
            elif rand_val < 0.4:
                # --- EASY LEGITIMATE PATTERNS ---
                # Low entropy, clean words, standard TLDs, no IPs
                
                # Core length features
                url_len = np.random.randint(20, 60)        # Shorter
                dom_len = np.random.randint(5, 20)
                path_len = max(0, url_len - dom_len - 8)
                
                # Complexity
                dots = np.random.randint(1, 3)             # google.com
                hyphens = np.random.randint(0, 2)          # rare hyphens
                slashes = np.random.randint(1, 4)          # shallow paths
                
                # Suspicious indicators
                has_ip = 0
                suspicious_tld = 0
                shortener = 0
                
                # New advanced features simulation
                entropy = np.random.uniform(1.5, 3.2)      # Low entropy (dictionary words)
                digit_ratio = np.random.uniform(0.0, 0.05) # Few digits
                longest_token = np.random.randint(3, 10)   # Short words
                tld_in_path = 0
                client_server = 0
                
                label = 0
                
            else:
                # --- HARD LEGITIMATE PATTERNS (The "College Website" Logic) ---
                # Academic/Corporate sites: Deep paths, subdomains, 'login' keyword, BUT trusted TLDs
                
                # Core length features
                url_len = np.random.randint(50, 100)       # Long (like phishing)
                dom_len = np.random.randint(15, 30)        # medium domain
                path_len = url_len - dom_len - 8
                
                # Complexity
                dots = np.random.randint(2, 5)             # portal.stanford.edu (3 dots)
                hyphens = np.random.randint(0, 3)          # some hyphens allowed
                slashes = np.random.randint(2, 6)          # /dept/student/login
                
                # Suspicious indicators
                has_ip = 0
                suspicious_tld = 0                         # Trusted TLDs usually
                shortener = 0
                
                # New advanced features simulation
                entropy = np.random.uniform(2.5, 3.8)      # Medium entropy (structured names)
                digit_ratio = np.random.uniform(0.0, 0.1)  # Some IDs allowed
                longest_token = np.random.randint(5, 15)   # 'administration'
                tld_in_path = 0
                client_server = np.random.choice([0, 1], p=[0.6, 0.4]) # 'server' might appear in tech blogs
                
                label = 0

            # Construct feature vector based on names order
            features = {
                'url_length': float(url_len),
                'domain_length': float(dom_len),
                'path_length': float(path_len),
                'num_dots': float(dots),
                'num_hyphens': float(hyphens),
                'num_slashes': float(slashes),
                # Randomize minor features
                'num_question_marks': float(np.random.randint(0, 2)),
                'num_equals': float(np.random.randint(0, 3)),
                'num_ampersands': float(np.random.randint(0, 2)),
                'num_percent': float(np.random.randint(0, 4) if label == 1 else 0),
                'has_ip': float(has_ip),
                'has_port': float(np.random.choice([0, 1], p=[0.95, 0.05]) if label == 1 else 0),
                # Hard Legitimate often has 'login' or 'account' keywords!
                'suspicious_keyword_count': float(np.random.randint(1, 3) if (label == 1 or rand_val >= 0.4) else 0),
                # Hard Legitimate IS trusted (simulating the fix)
                'is_trusted_domain': float(1.0 if (label == 0 and rand_val >= 0.4) else (1.0 if label == 0 and np.random.random() > 0.7 else 0.0)),
                'has_homograph': float(0.0),
                'subdomain_count': float(dots - 1),
                'path_depth': float(slashes),
                
                # NEW FEATURES
                'entropy': entropy,
                'digit_ratio': digit_ratio,
                'longest_token_len': float(longest_token),
                'tld_in_path': float(tld_in_path),
                'is_shortened': float(shortener),
                'is_suspicious_tld': float(suspicious_tld),
                'has_client_server': float(client_server),
                'domain_token_count': float(dots + 1),
                'path_token_count': float(slashes + 1),
                'is_punycode': 0.0
            }
            
            # Pack into list in correct order
            vector = [features[name] for name in self.feature_names]
            X.append(vector)
            y.append(label)
            
        print(f"[OK] Generated {len(X)} samples with {len(self.feature_names)} features each.")
        return np.array(X), np.array(y)

    def train(self):
        """Train the Random Forest model."""
        print("="*70)
        print("FINAL RANDOM FOREST TRAINING (High Accuracy Mode)")
        print("="*70)
        
        # 1. Generate Data
        X, y = self.create_enhanced_synthetic_data(n_samples=50000)
        
        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Initialize Random Forest with requested 'only Random Forest' constraint
        # but optimized for the larger dataset
        print("\n[*] Initializing Random Forest with 1000 trees...")
        self.model = RandomForestClassifier(
            n_estimators=1000,      # High number of trees for stability
            max_depth=None,         # Allow deep trees for complex feature interactions
            min_samples_split=2,    # Standard
            min_samples_leaf=1,     # Standard
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,              # Parallel processing
            random_state=42,
            verbose=1
        )
        
        # 4. Train
        print(f"[*] Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # 5. Evaluate
        print("\n[*] Evaluating...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("-" * 30)
        print(f"ACCURACY: {acc:.2%}")
        print("-" * 30)
        
        # Detailed report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # 6. Save
        self.save_model()
        
    def save_model(self):
        """Save the model and metadata."""
        if self.model:
            # Save model
            model_path = 'ml/phishing_model.pkl'
            print(f"\n[*] Saving model to {model_path}...")
            joblib.dump(self.model, model_path)
            
            # Save metadata for the predictor to use
            metadata = {
                'accuracy': self.model.score(self.create_enhanced_synthetic_data(1000)[0], np.array([0]*500 + [1]*500)), # Approx
                'feature_names': self.feature_names,
                'model_type': 'random_forest_50k'
            }
            with open('ml/model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            print("[*] Metadata saved.")
            print("\nTRAINING COMPLETE!")

if __name__ == '__main__':
    trainer = FinalRandomForestTrainer()
    trainer.train()
