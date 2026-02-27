# -*- coding: utf-8 -*-
"""
Train ML model using OpenPhish live phishing feed + existing datasets.

This script:
1. Downloads the latest phishing URLs from OpenPhish (feed.txt)
2. Combines with legitimate URLs for class balance
3. Retrains the Gradient Boosting model
4. Saves as the primary model (phishing_model_gb_optimized.pkl)

Usage:
    python ml/train_with_openphish.py
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.multi_dataset_loader import MultiDatasetLoader
from ml.improved_trainer import ImprovedPhishingTrainer


def main():
    print("\n" + "="*70)
    print("PHISHGUARD — OPENPHISH DATASET TRAINING")
    print("Downloads live phishing URLs from openphish.com/feed.txt")
    print("="*70)

    # Delete cached OpenPhish file to always get the freshest data
    cache_file = 'ml/datasets/openphish_dataset.csv'
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("[*] Cleared old OpenPhish cache — fetching fresh data...")

    loader = MultiDatasetLoader()

    print("\n[*] Loading datasets...")
    print("    - OpenPhish live feed (https://openphish.com/feed.txt)")
    print("    - Mendeley dataset (if available)")
    print("    - Legitimate URLs (auto-generated for balance)")

    try:
        X, y = loader.load_and_combine_datasets(
            use_phiusiil=False,      # Skip — large download
            use_phishtank=False,     # Skip — requires API key
            use_mendeley=True,       # Use if present
            use_openphish=True,      # ✅ Main source
            balance_classes=True
        )

        print(f"\n[SUCCESS] Dataset ready: {len(X)} samples, {X.shape[1]} features")
        print(f"  Legitimate: {(y == 0).sum()}")
        print(f"  Phishing:   {(y == 1).sum()}")

    except Exception as e:
        print(f"\n[ERROR] Failed to load datasets: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Train Gradient Boosting model
    print("\n" + "="*70)
    print("TRAINING GRADIENT BOOSTING MODEL")
    print("="*70)

    trainer = ImprovedPhishingTrainer()
    trainer.train_optimized_model(
        X=X,
        y=y,
        model_type='gradient_boosting',
        use_scaling=False,
        test_size=0.2,
        n_folds=5
    )

    # Save as the primary model used by predict.py
    trainer.save_model(
        model_path='ml/phishing_model_gb_optimized.pkl',
        metadata_path='ml/model_metadata_gb_optimized.json'
    )

    meta = trainer.training_metadata
    print("\n" + "="*70)
    print("TRAINING COMPLETE — MODEL SAVED")
    print("="*70)
    print(f"  Model:     ml/phishing_model_gb_optimized.pkl")
    print(f"  Accuracy:  {meta.get('accuracy', 0):.2%}")
    print(f"  Precision: {meta.get('precision', 0):.2%}")
    print(f"  Recall:    {meta.get('recall', 0):.2%}")
    print(f"  F1-Score:  {meta.get('f1_score', 0):.2%}")
    print(f"  ROC-AUC:   {meta.get('roc_auc', 0):.4f}")
    print(f"  Samples:   {len(X)}")
    print(f"  Features:  {X.shape[1]}")
    print("\n[NEXT] Restart Flask server to load new model:")
    print("       python app.py\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
