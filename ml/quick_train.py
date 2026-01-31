"""
Quick Training Script - Tests multi-dataset loader with fallback
This script trains the model with a smaller dataset for quick testing
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.trainer import PhishingModelTrainer
from ml.multi_dataset_loader import MultiDatasetLoader

def main():
    print("="*70)
    print("PHISHING GUARD - QUICK TRAINING TEST")
    print("="*70)
    
    trainer = PhishingModelTrainer()
    loader = MultiDatasetLoader()
    
    print("\n[*] Loading datasets (will use fallback if downloads fail)...")
    
    try:
        # Try to load datasets but with short timeout
        X, y = loader.load_and_combine_datasets(
            use_phiusiil=False,  # Skip large UCI dataset for quick test
            use_phishtank=True,  # Try PhishTank
            balance_classes=True
        )
        
        if len(X) < 100:
            print(f"[!] Only {len(X)} samples loaded, using synthetic data instead...")
            X, y = trainer.create_sample_data(n_samples=10000)
        else:
            print(f"[+] Loaded {len(X)} real samples")
            
    except Exception as e:
        print(f"[!] Error loading real data: {e}")
        print("[*] Using synthetic data...")
        X, y = trainer.create_sample_data(n_samples=10000)
    
    print(f"\n[*] Training with {len(X)} samples...")
    
    # Train with cross-validation
    trainer.train_with_cross_validation(
        X=X,
        y=y,
        model_type='random_forest',
        n_folds=5,
        test_size=0.2
    )
    
    # Save model
    print("\n[*] Saving model...")
    trainer.save_model()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nAccuracy: {trainer.training_metadata.get('accuracy', 0):.2%}")
    print(f"Precision: {trainer.training_metadata.get('precision', 0):.2%}")
    print(f"Recall: {trainer.training_metadata.get('recall', 0):.2%}")
    print(f"F1-Score: {trainer.training_metadata.get('f1_score', 0):.2%}")
    
    if 'cv_mean' in trainer.training_metadata:
        print(f"CV Mean: {trainer.training_metadata['cv_mean']:.2%}")
    
    print("="*70)


if __name__ == '__main__':
    main()
