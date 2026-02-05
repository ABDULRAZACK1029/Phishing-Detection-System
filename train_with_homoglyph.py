"""
Simple training script for homoglyph detection model
Generates synthetic data with all 28 features including has_ascii_homoglyph
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from datetime import datetime

from ml.feature_extraction import FeatureExtractor

def generate_training_data(n_samples=10000):
    """Generate synthetic training data with all 28 features."""
    print(f"Generating {n_samples} synthetic samples...")
    
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    
    X = []
    y = []
    
    np.random.seed(42)
    
    for i in range(n_samples):
        is_phishing = i >= n_samples // 2
        
        if is_phishing:
            # Phishing features
            features = {
                'url_length': float(np.random.randint(60, 250)),
                'domain_length': float(np.random.randint(15, 60)),
                'path_length': float(np.random.randint(20, 150)),
                'num_dots': float(np.random.randint(3, 8)),
                'num_hyphens': float(np.random.randint(2, 10)),
                'num_slashes': float(np.random.randint(4, 15)),
                'num_question_marks': float(np.random.randint(0, 3)),
                'num_equals': float(np.random.randint(0, 8)),
                'num_ampersands': float(np.random.randint(0, 6)),
                'num_percent': float(np.random.randint(0, 12)),
                'has_ip': float(np.random.choice([0, 1], p=[0.6, 0.4])),
                'has_port': float(np.random.choice([0, 1], p=[0.85, 0.15])),
                'suspicious_keyword_count': float(np.random.randint(2, 8)),
                'is_trusted_domain': 0.0,
                'has_homograph': float(np.random.choice([0, 1], p=[0.85, 0.15])),
                'subdomain_count': float(np.random.randint(2, 6)),
                'path_depth': float(np.random.randint(2, 8)),
                'entropy': float(np.random.uniform(3.0, 4.5)),
                'digit_ratio': float(np.random.uniform(0.1, 0.3)),
                'longest_token_len': float(np.random.randint(10, 30)),
                'tld_in_path': float(np.random.choice([0, 1], p=[0.7, 0.3])),
                'is_shortened': float(np.random.choice([0, 1], p=[0.9, 0.1])),
                'is_suspicious_tld': float(np.random.choice([0, 1], p=[0.7, 0.3])),
                'has_client_server': float(np.random.choice([0, 1], p=[0.8, 0.2])),
                'domain_token_count': float(np.random.randint(3, 8)),
                'path_token_count': float(np.random.randint(2, 10)),
                'is_punycode': float(np.random.choice([0, 1], p=[0.95, 0.05])),
                'has_ascii_homoglyph': float(np.random.choice([0, 1], p=[0.7, 0.3])),  # NEW
            }
            label = 1
        else:
            # Legitimate features
            features = {
                'url_length': float(np.random.randint(15, 80)),
                'domain_length': float(np.random.randint(8, 25)),
                'path_length': float(np.random.randint(0, 50)),
                'num_dots': float(np.random.randint(1, 3)),
                'num_hyphens': float(np.random.randint(0, 2)),
                'num_slashes': float(np.random.randint(2, 6)),
                'num_question_marks': float(np.random.randint(0, 2)),
                'num_equals': float(np.random.randint(0, 4)),
                'num_ampersands': float(np.random.randint(0, 3)),
                'num_percent': float(np.random.randint(0, 5)),
                'has_ip': float(np.random.choice([0, 1], p=[0.98, 0.02])),
                'has_port': float(np.random.choice([0, 1], p=[0.99, 0.01])),
                'suspicious_keyword_count': float(np.random.randint(0, 2)),
                'is_trusted_domain': float(np.random.choice([0, 1], p=[0.7, 0.3])),
                'has_homograph': 0.0,
                'subdomain_count': float(np.random.randint(0, 2)),
                'path_depth': float(np.random.randint(0, 4)),
                'entropy': float(np.random.uniform(2.5, 3.5)),
                'digit_ratio': float(np.random.uniform(0.0, 0.1)),
                'longest_token_len': float(np.random.randint(5, 15)),
                'tld_in_path': float(np.random.choice([0, 1], p=[0.95, 0.05])),
                'is_shortened': 0.0,
                'is_suspicious_tld': 0.0,
                'has_client_server': float(np.random.choice([0, 1], p=[0.95, 0.05])),
                'domain_token_count': float(np.random.randint(2, 4)),
                'path_token_count': float(np.random.randint(0, 5)),
                'is_punycode': 0.0,
                'has_ascii_homoglyph': 0.0,  # NEW - legitimate sites don't use homoglyphs
            }
            label = 0
        
        # Convert to feature vector
        feature_vector = [features[name] for name in feature_names]
        X.append(feature_vector)
        y.append(label)
    
    return np.array(X), np.array(y)


def train_model():
    """Train the Random Forest model with homoglyph detection."""
    print("=" * 70)
    print("PHISHING DETECTION MODEL TRAINING - WITH HOMOGLYPH DETECTION")
    print("=" * 70)
    print()
    
    # Generate training data
    X, y = generate_training_data(n_samples=10000)
    print(f"[OK] Generated {len(X)} samples")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    print()
    
    # Train Random Forest
    print("[*] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("[OK] Training completed")
    print()
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print()
    print(f"Accuracy: {accuracy:.2%}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Save model
    print("=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    print()
    
    joblib.dump(model, 'ml/phishing_model.pkl')
    print("[OK] Model saved to: ml/phishing_model.pkl")
    
    # Save metadata
    extractor = FeatureExtractor()
    metadata = {
        'model_type': 'random_forest',
        'accuracy': float(accuracy),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'total_samples': len(X),
        'feature_count': X.shape[1],
        'feature_names': extractor.get_feature_names(),
        'training_date': datetime.now().isoformat(),
        'includes_homoglyph_detection': True
    }
    
    with open('ml/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("[OK] Metadata saved to: ml/model_metadata.json")
    print()
    
    print("=" * 70)
    print("[SUCCESS] Training completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Test homoglyph detection: py test_homoglyph_detection.py")
    print("  2. Start Flask app: py app.py")
    print()


if __name__ == '__main__':
    train_model()
