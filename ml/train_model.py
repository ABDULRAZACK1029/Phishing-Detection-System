# -*- coding: utf-8 -*-
"""
Phishing Model Training Script
Trains improved model with better generalization
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.trainer import PhishingModelTrainer

def main():
    """Train the phishing detection model with improved hyperparameters."""
    
    print("="*70)
    print("PHISHING GUARD - ML MODEL TRAINING")
    print("="*70)
    
    # Initialize trainer
    trainer = PhishingModelTrainer()
    
    print("\n[*] Generating enhanced training dataset...")
    print("[*] Using 15,000 synthetic samples with realistic patterns")
    
    # Create larger, more diverse synthetic dataset
    X, y = trainer.create_sample_data(n_samples=15000)
    
    print(f"\n[+] Dataset created: {len(X)} samples")
    print(f"    Legitimate: {sum(y == 0)}")
    print(f"    Phishing: {sum(y == 1)}")
    
    print("\n[*] Training Random Forest with cross-validation...")
    print("[*] Using improved hyperparameters for better generalization")
    
    # Train with cross-validation
    trainer.train_with_cross_validation(
        X=X,
        y=y,
        model_type='random_forest',
        n_folds=5,
        test_size=0.2
    )
    
    # Save model
    print("\n[*] Saving model and metadata...")
    trainer.save_model()
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    metadata = trainer.training_metadata
    
    print("\n[PERFORMANCE METRICS]")
    print(f"  Test Accuracy:     {metadata.get('accuracy', 0):.2%}")
    print(f"  Training Accuracy: {metadata.get('train_accuracy', 0):.2%}")
    print(f"  Precision:         {metadata.get('precision', 0):.2%}")
    print(f"  Recall:            {metadata.get('recall', 0):.2%}")
    print(f"  F1-Score:          {metadata.get('f1_score', 0):.2%}")
    
    if 'roc_auc' in metadata and metadata['roc_auc']:
        print(f"  ROC-AUC:           {metadata['roc_auc']:.4f}")
    
    print("\n[CROSS-VALIDATION]")
    if 'cv_mean' in metadata:
        print(f"  Mean CV Accuracy:  {metadata['cv_mean']:.2%}")
        print(f"  Std Deviation:     {metadata['cv_std']:.2%}")
        
        # Check overfitting
        train_test_gap = metadata.get('train_accuracy', 0) - metadata.get('accuracy', 0)
        if train_test_gap > 0.05:
            print(f"\n  WARNING: Train-test gap is {train_test_gap:.2%} (possible overfitting)")
        else:
            print(f"\n  GOOD: Train-test gap is {train_test_gap:.2%} (good generalization)")
    
    print("\n[DATASET INFO]")
    print(f"  Total Samples:     {metadata.get('total_samples', 'N/A')}")
    print(f"  Training Set:      {metadata.get('training_samples', 'N/A')}")
    print(f"  Test Set:          {metadata.get('test_samples', 'N/A')}")
    print(f"  Features:          {metadata.get('feature_count', 'N/A')}")
    
    print("\n[MODEL CONFIGURATION]")
    print(f"  Model Type:        Random Forest")
    print(f"  Trees:             300")
    print(f"  Max Depth:         20")
    print(f"  Min Samples Split: 10")
    print(f"  Class Weight:      Balanced")
    
    print("\n" + "="*70)
    print("Model saved to: ml/phishing_model.pkl")
    print("Metadata saved to: ml/model_metadata.json")
    print("="*70)
    
    print("\n[NEXT STEPS]")
    print("  1. Test predictions: py ml/predict.py")
    print("  2. Start Flask app: py app.py")
    print("  3. Test in browser: http://localhost:5000")
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
