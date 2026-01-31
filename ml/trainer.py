"""
ML Model Trainer for Phishing Detection
Trains scikit-learn models for phishing URL detection

This module provides a production-ready training pipeline suitable for
BCA Final Year project. Supports Random Forest and Logistic Regression.

For production use with real data:
1. Download phishing dataset from PhishTank (https://www.phishtank.com/)
2. Or use UCI ML Repository phishing dataset
3. Replace create_sample_data() with load_real_data()
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import joblib
import json
import os
from datetime import datetime
from typing import Tuple, Dict, Any

# Import feature extractor
try:
    from ml.feature_extraction import FeatureExtractor
except ImportError:
    from feature_extraction import FeatureExtractor


class PhishingModelTrainer:
    """
    Trainer for phishing detection ML models.
    
    This class implements a complete training pipeline including:
    - Synthetic data generation (for development/testing)
    - Support for real dataset loading
    - Multiple model architectures (Random Forest, Logistic Regression)
    - Comprehensive evaluation metrics
    - Model serialization and metadata storage
    
    Usage:
        trainer = PhishingModelTrainer()
        model = trainer.train(model_type='random_forest')
        trainer.save_model()
    """
    
    def __init__(self):
        """Initialize the trainer with feature extractor and default settings."""
        self.model = None
        self.model_type = None
        self.feature_extractor = FeatureExtractor()
        self.feature_names = self.feature_extractor.get_feature_names()
        self.training_metadata = {}
    
    def create_sample_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic training data for development and testing.
        
        In production, replace this with real phishing dataset from:
        - PhishTank API: https://www.phishtank.com/developer_info.php
        - UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/phishing+websites
        - Kaggle Phishing Datasets
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix X and labels y
        """
        print(f"Generating {n_samples} synthetic samples...")
        
        X = []
        y = []
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_samples):
            # Create two types of samples: legitimate (50%) and phishing (50%)
            is_phishing = i >= n_samples // 2
            
            if is_phishing:
                # Generate phishing-like features
                features = {
                    'url_length': np.random.randint(60, 250),  # Longer URLs
                    'domain_length': np.random.randint(15, 60),
                    'path_length': np.random.randint(20, 150),
                    'num_dots': np.random.randint(3, 8),  # More dots
                    'num_hyphens': np.random.randint(2, 10),  # More hyphens
                    'num_slashes': np.random.randint(4, 15),
                    'num_question_marks': np.random.randint(0, 3),
                    'num_equals': np.random.randint(0, 8),
                    'num_ampersands': np.random.randint(0, 6),
                    'num_percent': np.random.randint(0, 12),
                    'has_ip': np.random.choice([0, 1], p=[0.6, 0.4]),  # Higher chance of IP
                    'has_port': np.random.choice([0, 1], p=[0.85, 0.15]),
                    'suspicious_keyword_count': np.random.randint(2, 8),  # More keywords
                    'is_trusted_domain': 0,  # Not trusted
                    'has_homograph': np.random.choice([0, 1], p=[0.85, 0.15]),
                    'subdomain_count': np.random.randint(2, 6),  # More subdomains
                    'path_depth': np.random.randint(2, 8),
                }
                label = 1  # Phishing
            else:
                # Generate legitimate-like features
                features = {
                    'url_length': np.random.randint(15, 80),  # Shorter URLs
                    'domain_length': np.random.randint(8, 25),
                    'path_length': np.random.randint(0, 50),
                    'num_dots': np.random.randint(1, 3),  # Fewer dots
                    'num_hyphens': np.random.randint(0, 2),  # Fewer hyphens
                    'num_slashes': np.random.randint(2, 6),
                    'num_question_marks': np.random.randint(0, 2),
                    'num_equals': np.random.randint(0, 4),
                    'num_ampersands': np.random.randint(0, 3),
                    'num_percent': np.random.randint(0, 5),
                    'has_ip': np.random.choice([0, 1], p=[0.98, 0.02]),  # Rare IP usage
                    'has_port': np.random.choice([0, 1], p=[0.99, 0.01]),
                    'suspicious_keyword_count': np.random.randint(0, 2),  # Fewer keywords
                    'is_trusted_domain': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'has_homograph': 0,  # No homograph
                    'subdomain_count': np.random.randint(0, 2),  # Fewer subdomains
                    'path_depth': np.random.randint(0, 4),
                }
                label = 0  # Legitimate
            
            # Convert to feature vector in correct order
            feature_vector = [features[name] for name in self.feature_names]
            X.append(feature_vector)
            y.append(label)
        
        print(f"[OK] Generated {n_samples} samples (Legitimate: {sum(1 for label in y if label == 0)}, Phishing: {sum(y)})")
        
        return np.array(X), np.array(y)
    
    def load_real_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load real phishing dataset from CSV file.
        
        Expected CSV format:
        - Columns matching feature names from feature_extractor
        - 'label' column with 0 (legitimate) or 1 (phishing)
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix X and labels y
        """
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Extract features and labels
        X = df[self.feature_names].values
        y = df['label'].values
        
        print(f"[OK] Loaded {len(X)} samples from {filepath}")
        return X, y
    
    def train(
        self, 
        X: np.ndarray = None, 
        y: np.ndarray = None,
        model_type: str = 'random_forest',
        test_size: float = 0.2
    ) -> Any:
        """
        Train the phishing detection model.
        
        Args:
            X (np.ndarray): Feature matrix (if None, generates synthetic data)
            y (np.ndarray): Labels (if None, generates synthetic data)
            model_type (str): 'random_forest' or 'logistic_regression'
            test_size (float): Proportion of data for testing
            
        Returns:
            Trained model object
        """
        print("\n" + "="*70)
        print("PHISHING DETECTION MODEL TRAINING")
        print("="*70)
        
        # Generate or use provided data
        if X is None or y is None:
            X, y = self.create_sample_data(n_samples=5000)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nDataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Features: {len(self.feature_names)}")
        
        # Initialize model based on type
        self.model_type = model_type
        
        if model_type == 'random_forest':
            print("\n[*] Training Random Forest Classifier...")
            self.model = RandomForestClassifier(
                n_estimators=300,          # Increased trees for better stability
                max_depth=20,              # Deeper trees for complex patterns
                min_samples_split=10,      # Prevent overfitting on noise
                min_samples_leaf=4,        # Larger leaves for generalization
                max_features='sqrt',       # Random feature selection
                class_weight='balanced',   # Handle class imbalance
                random_state=42,           # Reproducibility
                n_jobs=-1                  # Use all CPU cores
            )
        elif model_type == 'logistic_regression':
            print("\n[*] Training Logistic Regression...")
            self.model = LogisticRegression(
                max_iter=1000,         # Maximum iterations
                C=1.0,                 # Regularization strength
                random_state=42,       # Reproducibility
                n_jobs=-1              # Use all CPU cores
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'random_forest' or 'logistic_regression'")
        
        # Train the model
        self.model.fit(X_train, y_train)
        print("[OK] Training completed")
        
        # Evaluate on test set
        print("\n" + "-"*70)
        print("MODEL EVALUATION")
        print("-"*70)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[*] Overall Accuracy: {accuracy:.2%}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n[*] Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                Legit  Phishing")
        print(f"Actual  Legit    {cm[0][0]:5d}    {cm[0][1]:5d}")
        print(f"        Phishing {cm[1][0]:5d}    {cm[1][1]:5d}")
        
        # Classification Report
        print(f"\n[*] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Feature Importance (Random Forest only)
        if model_type == 'random_forest':
            print(f"\n[*] Top 10 Most Important Features:")
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            for i, idx in enumerate(indices, 1):
                print(f"   {i:2d}. {self.feature_names[idx]:25s} - {importances[idx]:.4f}")
        
        # Store metadata
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        # Calculate ROC-AUC if model supports probability predictions
        try:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = None
        
        self.training_metadata = {
            'model_type': model_type,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X),
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'confusion_matrix': cm.tolist(),
            'test_size': test_size
        }
        
        print("\n" + "="*70)
        print("[SUCCESS] Training completed successfully!")
        print("="*70 + "\n")
        
        return self.model
    
    def train_with_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'random_forest',
        n_folds: int = 5,
        test_size: float = 0.2
    ) -> Any:
        """
        Train with cross-validation for more robust evaluation.
        
        This method performs k-fold cross-validation during training
        to ensure the model generalizes well across different data splits.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            model_type (str): 'random_forest' or 'logistic_regression'
            n_folds (int): Number of cross-validation folds
            test_size (float): Proportion of data for final testing
            
        Returns:
            Trained model object
        """
        print("\n" + "="*70)
        print("PHISHING DETECTION MODEL TRAINING (WITH CROSS-VALIDATION)")
        print("="*70)
        
        # Split data into training and testing sets (for final evaluation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nDataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Features: {len(self.feature_names)}")
        
        # Initialize model based on type
        self.model_type = model_type
        
        if model_type == 'random_forest':
            print(f"\n[*] Training Random Forest Classifier with {n_folds}-Fold Cross-Validation...")
            self.model = RandomForestClassifier(
                n_estimators=300,          # Increased trees for better stability
                max_depth=20,              # Deeper trees for complex patterns
                min_samples_split=10,      # Prevent overfitting on noise
                min_samples_leaf=4,        # Larger leaves for generalization
                max_features='sqrt',       # Random feature selection
                class_weight='balanced',   # Handle class imbalance
                random_state=42,           # Reproducibility
                n_jobs=-1                  # Use all CPU cores
            )
        elif model_type == 'logistic_regression':
            print(f"\n[*] Training Logistic Regression with {n_folds}-Fold Cross-Validation...")
            self.model = LogisticRegression(
                max_iter=1000,         # Maximum iterations
                C=1.0,                 # Regularization strength
                random_state=42,       # Reproducibility
                n_jobs=-1              # Use all CPU cores
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform cross-validation on training data
        print(f"\n[*] Performing {n_folds}-Fold Cross-Validation...")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"\n[*] Cross-Validation Results:")
        print(f"  Fold Accuracies: {[f'{score:.2%}' for score in cv_scores]}")
        print(f"  Mean Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
        
        # Train final model on full training set
        print(f"\n[*] Training final model on full training set...")
        self.model.fit(X_train, y_train)
        print("[OK] Training completed")
        
        # Evaluate on held-out test set
        print("\n" + "-"*70)
        print("MODEL EVALUATION (HELD-OUT TEST SET)")
        print("-"*70)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[*] Test Set Accuracy: {accuracy:.2%}")
        
        # Check for overfitting
        train_accuracy = self.model.score(X_train, y_train)
        print(f"[*] Training Set Accuracy: {train_accuracy:.2%}")
        if train_accuracy - accuracy > 0.05:
            print(f"[WARNING]  Warning: Possible overfitting detected (train-test gap: {(train_accuracy - accuracy):.2%})")
        else:
            print(f"[SUCCESS] Good generalization (train-test gap: {(train_accuracy - accuracy):.2%})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n[*] Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                Legit  Phishing")
        print(f"Actual  Legit    {cm[0][0]:5d}    {cm[0][1]:5d}")
        print(f"        Phishing {cm[1][0]:5d}    {cm[1][1]:5d}")
        
        # Classification Report
        print(f"\n[*] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Feature Importance (Random Forest only)
        if model_type == 'random_forest':
            print(f"\n[*] Top 10 Most Important Features:")
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            for i, idx in enumerate(indices, 1):
                print(f"   {i:2d}. {self.feature_names[idx]:25s} - {importances[idx]:.4f}")
        
        # Store metadata
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        # Calculate ROC-AUC
        try:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\n[*] ROC-AUC Score: {roc_auc:.4f}")
        except:
            roc_auc = None
        
        self.training_metadata = {
            'model_type': model_type,
            'accuracy': float(accuracy),
            'train_accuracy': float(train_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'cross_validation_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'n_folds': n_folds,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X),
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'confusion_matrix': cm.tolist(),
            'test_size': test_size,
            'dataset_source': 'multi_dataset'
        }
        
        print("\n" + "="*70)
        print("[SUCCESS] Training with cross-validation completed successfully!")
        print("="*70 + "\n")
        
        return self.model
    
    
    def save_model(self, model_path: str = 'ml/phishing_model.pkl', 
                   metadata_path: str = 'ml/model_metadata.json'):
        """
        Save trained model and metadata to disk.
        
        Args:
            model_path (str): Path to save the model file
            metadata_path (str): Path to save metadata JSON
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first using train()")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model using joblib (more efficient than pickle for sklearn)
        joblib.dump(self.model, model_path)
        print(f"[OK] Model saved to: {model_path}")
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        print(f"[OK] Metadata saved to: {metadata_path}")
    
    def load_model(self, model_path: str = 'ml/phishing_model.pkl') -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model file
            
        Returns:
            Loaded model object
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"[OK] Model loaded from: {model_path}")
        
        return self.model


def main():
    """Main function to train and save the phishing detection model with multiple datasets."""
    
    print("\n" + "="*70)
    print("PHISHING GUARD - ML MODEL TRAINING")
    print("Multi-Dataset Approach for Improved Accuracy")
    print("="*70 + "\n")
    
    # Import multi-dataset loader
    try:
        from ml.multi_dataset_loader import MultiDatasetLoader
    except ImportError:
        from multi_dataset_loader import MultiDatasetLoader
    
    # Initialize trainer
    trainer = PhishingModelTrainer()
    
    # Load multiple datasets
    print("📦 Loading datasets from multiple sources...")
    loader = MultiDatasetLoader()
    
    try:
        # Attempt to load real datasets (PhiUSIIL + PhishTank + Legitimate)
        X, y = loader.load_and_combine_datasets(
            use_phiusiil=True,
            use_phishtank=True,
            balance_classes=True
        )
        
        print(f"[SUCCESS] Successfully loaded {len(X)} samples")
        
        # Train with cross-validation for robust evaluation
        print("\n[*] Training Random Forest model with 5-fold cross-validation...")
        trainer.train_with_cross_validation(
            X=X,
            y=y,
            model_type='random_forest',
            n_folds=5,
            test_size=0.2
        )
        
    except Exception as e:
        print(f"\n[WARNING]  Error loading real datasets: {e}")
        print("📝 Falling back to synthetic data for training...\n")
        
        # Fallback to synthetic data
        trainer.train(model_type='random_forest')
    
    # Save the model
    print("\n💾 Saving model...")
    trainer.save_model()
    
    print("\n" + "="*70)
    print("[DONE] Model training and saving completed!")
    print("="*70)
    print("\n[*] Model Performance Summary:")
    print(f"  Accuracy: {trainer.training_metadata.get('accuracy', 0):.2%}")
    print(f"  Precision: {trainer.training_metadata.get('precision', 0):.2%}")
    print(f"  Recall: {trainer.training_metadata.get('recall', 0):.2%}")
    print(f"  F1-Score: {trainer.training_metadata.get('f1_score', 0):.2%}")
    
    if 'roc_auc' in trainer.training_metadata and trainer.training_metadata['roc_auc']:
        print(f"  ROC-AUC: {trainer.training_metadata['roc_auc']:.4f}")
    
    if 'cv_mean' in trainer.training_metadata:
        print(f"\n[*] Cross-Validation:")
        print(f"  Mean CV Accuracy: {trainer.training_metadata['cv_mean']:.2%}")
        print(f"  CV Std Dev: {trainer.training_metadata['cv_std']:.2%}")
    
    print(f"\n📁 Training Data:")
    print(f"  Total samples: {trainer.training_metadata.get('total_samples', 'N/A')}")
    print(f"  Training: {trainer.training_metadata.get('training_samples', 'N/A')}")
    print(f"  Testing: {trainer.training_metadata.get('test_samples', 'N/A')}")
    
    print("\n✨ Next steps:")
    print("  1. Test the model: python ml/predict.py")
    print("  2. Start Flask app: python app.py")
    print("  3. Test phishing detection on real URLs")
    print("="*70 + "\n")



if __name__ == '__main__':
    main()


