# -*- coding: utf-8 -*-
"""
Enhanced Phishing Model Training Script
Improved accuracy with optimized hyperparameters and ensemble methods
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import json
from datetime import datetime

from ml.trainer import PhishingModelTrainer
from ml.feature_extraction import FeatureExtractor


class ImprovedPhishingTrainer(PhishingModelTrainer):
    """
    Enhanced trainer with advanced ML techniques for higher accuracy.
    
    Improvements:
    1. Optimized Random Forest hyperparameters
    2. Gradient Boosting Classifier
    3. Ensemble voting (combines multiple models)
    4. Feature scaling
    5. Hyperparameter tuning via GridSearchCV
    """
    
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.best_params = None
    
    def get_optimized_random_forest(self):
        """
        Get Random Forest with research-backed optimal hyperparameters for phishing detection.
        
        These parameters are based on empirical research showing best performance
        for URL-based phishing detection tasks.
        """
        return RandomForestClassifier(
            n_estimators=500,           # More trees = better accuracy (diminishing returns after 500)
            max_depth=30,               # Deeper trees for complex patterns
            min_samples_split=5,        # Allow more granular splits
            min_samples_leaf=2,         # Smaller leaves = more detailed patterns
            max_features='sqrt',        # sqrt(n_features) for good balance
            class_weight='balanced',    # Handle class imbalance
            bootstrap=True,             # Bootstrap sampling
            oob_score=True,             # Out-of-bag score for validation
            random_state=42,
            n_jobs=-1,                  # Use all CPU cores
            verbose=0
        )
    
    def get_gradient_boosting(self):
        """
        Get Gradient Boosting Classifier - often outperforms Random Forest.
        
        Gradient Boosting builds trees sequentially, with each tree correcting
        errors from previous trees.
        """
        return GradientBoostingClassifier(
            n_estimators=300,           # Number of boosting stages
            learning_rate=0.1,          # Shrinks contribution of each tree
            max_depth=7,                # Depth of individual trees
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,              # Fraction of samples for fitting trees
            max_features='sqrt',
            random_state=42,
            verbose=0
        )
    
    def create_ensemble_model(self):
        """
        Create an ensemble model combining Random Forest and Gradient Boosting.
        
        Ensemble methods often achieve higher accuracy by combining predictions
        from multiple models.
        """
        rf = self.get_optimized_random_forest()
        gb = self.get_gradient_boosting()
        
        # Voting classifier - combines predictions with soft voting (averages probabilities)
        ensemble = VotingClassifier(
            estimators=[
                ('random_forest', rf),
                ('gradient_boosting', gb)
            ],
            voting='soft',  # Use probability estimates
            n_jobs=-1
        )
        
        return ensemble
    
    def train_optimized_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'ensemble',  # 'random_forest', 'gradient_boosting', or 'ensemble'
        use_scaling: bool = False,
        test_size: float = 0.2,
        n_folds: int = 5
    ):
        """
        Train model with optimized hyperparameters and cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            model_type: 'random_forest', 'gradient_boosting', or 'ensemble'
            use_scaling: Whether to scale features (recommended for some algorithms)
            test_size: Test set proportion
            n_folds: Number of CV folds
        """
        print("\n" + "="*70)
        print("ENHANCED PHISHING DETECTION MODEL TRAINING")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nDataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples:  {len(X_test)}")
        print(f"  Features:         {len(self.feature_names)}")
        
        # Feature scaling (optional but can help)
        if use_scaling:
            print("\n[*] Applying feature scaling...")
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Select model
        self.model_type = model_type
        
        if model_type == 'random_forest':
            print("\n[*] Training Optimized Random Forest...")
            self.model = self.get_optimized_random_forest()
            
        elif model_type == 'gradient_boosting':
            print("\n[*] Training Gradient Boosting Classifier...")
            self.model = self.get_gradient_boosting()
            
        elif model_type == 'ensemble':
            print("\n[*] Training Ensemble Model (Random Forest + Gradient Boosting)...")
            self.model = self.create_ensemble_model()
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Cross-validation
        print(f"\n[*] Performing {n_folds}-Fold Cross-Validation...")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"\n[*] Cross-Validation Results:")
        print(f"  Fold Accuracies: {[f'{score:.2%}' for score in cv_scores]}")
        print(f"  Mean Accuracy:   {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
        
        # Train final model
        print(f"\n[*] Training final model on full training set...")
        self.model.fit(X_train, y_train)
        print("[OK] Training completed")
        
        # Evaluate
        print("\n" + "-"*70)
        print("MODEL EVALUATION")
        print("-"*70)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = self.model.score(X_train, y_train)
        
        print(f"\n[*] Test Set Accuracy:     {accuracy:.2%}")
        print(f"[*] Training Set Accuracy: {train_accuracy:.2%}")
        
        # Check overfitting
        train_test_gap = train_accuracy - accuracy
        if train_test_gap > 0.05:
            print(f"[WARNING] Possible overfitting (gap: {train_test_gap:.2%})")
        else:
            print(f"[SUCCESS] Good generalization (gap: {train_test_gap:.2%})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n[*] Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                Legit  Phishing")
        print(f"Actual  Legit    {cm[0][0]:5d}    {cm[0][1]:5d}")
        print(f"        Phishing {cm[1][0]:5d}    {cm[1][1]:5d}")
        
        # Classification Report
        print(f"\n[*] Detailed Metrics:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # ROC-AUC
        try:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\n[*] ROC-AUC Score: {roc_auc:.4f}")
        except:
            roc_auc = None
        
        # Feature importance (if available)
        if model_type == 'random_forest':
            print(f"\n[*] Top 10 Most Important Features:")
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            for i, idx in enumerate(indices, 1):
                print(f"   {i:2d}. {self.feature_names[idx]:25s} - {importances[idx]:.4f}")
        
        # Store metadata
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
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
            'use_scaling': use_scaling,
            'dataset_source': 'enhanced_training'
        }
        
        print("\n" + "="*70)
        print("[SUCCESS] Enhanced training completed!")
        print("="*70 + "\n")
        
        return self.model
    
    def hyperparameter_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'random_forest',
        n_folds: int = 3
    ):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Warning: This can be time-consuming but finds optimal parameters.
        """
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING (This may take a while...)")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if model_type == 'random_forest':
            # Parameter grid for Random Forest
            param_grid = {
                'n_estimators': [300, 500],
                'max_depth': [20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            base_model = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
        elif model_type == 'gradient_boosting':
            # Parameter grid for Gradient Boosting
            param_grid = {
                'n_estimators': [200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [5, 7, 9],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            base_model = GradientBoostingClassifier(
                random_state=42
            )
        else:
            raise ValueError(f"Hyperparameter tuning not supported for: {model_type}")
        
        print(f"\n[*] Searching {len(param_grid)} hyperparameter combinations...")
        print(f"[*] Using {n_folds}-Fold Cross-Validation...")
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n[SUCCESS] Hyperparameter tuning completed!")
        print(f"\n[*] Best Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"    {param}: {value}")
        
        print(f"\n[*] Best CV Score: {grid_search.best_score_:.2%}")
        
        # Evaluate on test set
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        test_accuracy = self.model.score(X_test, y_test)
        print(f"[*] Test Set Accuracy: {test_accuracy:.2%}")
        
        return self.model
    
    def save_model(self, model_path: str = 'ml/phishing_model.pkl', 
                   metadata_path: str = 'ml/model_metadata.json'):
        """Save model with scaler if scaling was used."""
        super().save_model(model_path, metadata_path)
        
        # Save scaler separately if it was fitted
        if hasattr(self.scaler, 'mean_'):
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            print(f"[OK] Scaler saved to: {scaler_path}")


def main():
    """Train with optimized parameters for maximum accuracy."""
    
    print("\n" + "="*70)
    print("PHISHING GUARD - ENHANCED ML MODEL TRAINING")
    print("="*70)
    
    # Initialize trainer
    trainer = ImprovedPhishingTrainer()
    
    print("\n[*] Generating enhanced training dataset...")
    print("[*] Using 20,000 synthetic samples with realistic patterns")
    
    # Create larger, more diverse dataset
    X, y = trainer.create_sample_data(n_samples=20000)
    
    print(f"\n[+] Dataset created: {len(X)} samples")
    print(f"    Legitimate: {sum(y == 0)}")
    print(f"    Phishing:   {sum(y == 1)}")
    
    # Option 1: Train with optimized Random Forest
    print("\n" + "="*70)
    print("OPTION 1: OPTIMIZED RANDOM FOREST")
    print("="*70)
    
    trainer.train_optimized_model(
        X=X,
        y=y,
        model_type='random_forest',
        use_scaling=False,
        test_size=0.2,
        n_folds=5
    )
    
    print("\n[*] Saving optimized Random Forest model...")
    trainer.save_model(
        model_path='ml/phishing_model_rf.pkl',
        metadata_path='ml/model_metadata_rf.json'
    )
    
    # Option 2: Train Gradient Boosting
    print("\n" + "="*70)
    print("OPTION 2: GRADIENT BOOSTING")
    print("="*70)
    
    trainer2 = ImprovedPhishingTrainer()
    trainer2.train_optimized_model(
        X=X,
        y=y,
        model_type='gradient_boosting',
        use_scaling=False,
        test_size=0.2,
        n_folds=5
    )
    
    print("\n[*] Saving Gradient Boosting model...")
    trainer2.save_model(
        model_path='ml/phishing_model_gb.pkl',
        metadata_path='ml/model_metadata_gb.json'
    )
    
    # Option 3: Train Ensemble (Best accuracy)
    print("\n" + "="*70)
    print("OPTION 3: ENSEMBLE MODEL (RECOMMENDED)")
    print("="*70)
    
    trainer3 = ImprovedPhishingTrainer()
    trainer3.train_optimized_model(
        X=X,
        y=y,
        model_type='ensemble',
        use_scaling=False,
        test_size=0.2,
        n_folds=5
    )
    
    print("\n[*] Saving Ensemble model as primary model...")
    trainer3.save_model(
        model_path='ml/phishing_model.pkl',  # This will be the default model
        metadata_path='ml/model_metadata.json'
    )
    
    # Print comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    print(f"\nRandom Forest:")
    print(f"  Accuracy:  {trainer.training_metadata['accuracy']:.2%}")
    print(f"  Precision: {trainer.training_metadata['precision']:.2%}")
    print(f"  Recall:    {trainer.training_metadata['recall']:.2%}")
    print(f"  F1-Score:  {trainer.training_metadata['f1_score']:.2%}")
    
    print(f"\nGradient Boosting:")
    print(f"  Accuracy:  {trainer2.training_metadata['accuracy']:.2%}")
    print(f"  Precision: {trainer2.training_metadata['precision']:.2%}")
    print(f"  Recall:    {trainer2.training_metadata['recall']:.2%}")
    print(f"  F1-Score:  {trainer2.training_metadata['f1_score']:.2%}")
    
    print(f"\nEnsemble (Combined):")
    print(f"  Accuracy:  {trainer3.training_metadata['accuracy']:.2%}")
    print(f"  Precision: {trainer3.training_metadata['precision']:.2%}")
    print(f"  Recall:    {trainer3.training_metadata['recall']:.2%}")
    print(f"  F1-Score:  {trainer3.training_metadata['f1_score']:.2%}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    
    print("\n[NEXT STEPS]")
    print("  1. Test predictions: py ml/predict.py")
    print("  2. Start Flask app: py app.py")
    print("  3. Or train with real data using multi_dataset_loader.py")
    print("\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
