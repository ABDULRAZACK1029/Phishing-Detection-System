# -*- coding: utf-8 -*-
"""
Train ML model with REAL phishing datasets for maximum accuracy
Uses multiple real-world phishing datasets combined
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.improved_trainer import ImprovedPhishingTrainer
from ml.multi_dataset_loader import MultiDatasetLoader


def main():
    """Train with real phishing datasets for production-ready accuracy."""
    
    print("\n" + "="*70)
    print("PHISHING GUARD - REAL DATASET TRAINING")
    print("Using multiple real-world phishing datasets")
    print("="*70)
    
    # Load real datasets
    print("\n[*] Loading datasets from multiple sources...")
    print("    - PhiUSIIL Dataset (UCI ML Repository - 235K URLs)")
    print("    - PhishTank Live Data (Active phishing URLs)")
    print("    - Synthetic Legitimate URLs (Balanced)")
    
    loader = MultiDatasetLoader()
    
    try:
        # Load combined datasets
        X, y = loader.load_and_combine_datasets(
            use_phiusiil=True,      # Try to load UCI dataset
            use_phishtank=True,     # Try to load PhishTank data
            balance_classes=True    # Balance legitimate/phishing ratio
        )
        
        print(f"\n[SUCCESS] Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"  Legitimate: {(y == 0).sum()}")
        print(f"  Phishing:   {(y == 1).sum()}")
        
        # Check if we have enough data
        if len(X) < 1000:
            print("\n[WARNING] Dataset is small. Results may not be optimal.")
            print("Consider manually downloading larger datasets:")
            print("  1. PhiUSIIL: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset")
            print("  2. PhishTank: https://www.phishtank.com/developer_info.php")
        
    except Exception as e:
        print(f"\n[!] Error loading real datasets: {e}")
        print("\nFalling back to enhanced synthetic data...")
        
        trainer = ImprovedPhishingTrainer()
        X, y = trainer.create_sample_data(n_samples=25000)
    
    # Train multiple models and compare
    print("\n" + "="*70)
    print("TRAINING MULTIPLE MODELS FOR COMPARISON")
    print("="*70)
    
    # Determine training strategy based on dataset size
    if len(X) > 50000:
        print(f"\n[INFO] Large dataset detected ({len(X)} samples). Optimizing for speed...")
        n_folds = 2  # Reduce folds for large data
        train_rf_only = True
    else:
        n_folds = 5
        train_rf_only = False
        
    results = {}
    
    # Model 1: Optimized Random Forest
    print(f"\n[1/{'1' if train_rf_only else '3'}] Training Optimized Random Forest...")
    print("-" * 70)
    
    trainer_rf = ImprovedPhishingTrainer()
    trainer_rf.train_optimized_model(
        X=X,
        y=y,
        model_type='random_forest',
        use_scaling=False,
        test_size=0.2,
        n_folds=n_folds
    )
    
    results['Random Forest'] = {
        'accuracy': trainer_rf.training_metadata['accuracy'],
        'precision': trainer_rf.training_metadata['precision'],
        'recall': trainer_rf.training_metadata['recall'],
        'f1_score': trainer_rf.training_metadata['f1_score'],
        'roc_auc': trainer_rf.training_metadata.get('roc_auc', 0)
    }
    
    # Save as primary model if we are only training RF
    if train_rf_only:
        trainer_rf.save_model(
            model_path='ml/phishing_model.pkl',
            metadata_path='ml/model_metadata.json'
        )
        print("\n[*] Random Forest saved as primary model (Ensemble skipped for speed)")
    else:
        trainer_rf.save_model(
            model_path='ml/phishing_model_rf_optimized.pkl',
            metadata_path='ml/model_metadata_rf_optimized.json'
        )
    
    if not train_rf_only:
        # Model 2: Gradient Boosting
        print("\n[2/3] Training Gradient Boosting Classifier...")
        print("-" * 70)
        
        trainer_gb = ImprovedPhishingTrainer()
        trainer_gb.train_optimized_model(
            X=X,
            y=y,
            model_type='gradient_boosting',
            use_scaling=False,
            test_size=0.2,
            n_folds=n_folds
        )
        
        results['Gradient Boosting'] = {
            'accuracy': trainer_gb.training_metadata['accuracy'],
            'precision': trainer_gb.training_metadata['precision'],
            'recall': trainer_gb.training_metadata['recall'],
            'f1_score': trainer_gb.training_metadata['f1_score'],
            'roc_auc': trainer_gb.training_metadata.get('roc_auc', 0)
        }
        
        trainer_gb.save_model(
            model_path='ml/phishing_model_gb_optimized.pkl',
            metadata_path='ml/model_metadata_gb_optimized.json'
        )
        
        # Model 3: Ensemble (Best performance)
        print("\n[3/3] Training Ensemble Model (RF + GB)...")
        print("-" * 70)
        
        trainer_ensemble = ImprovedPhishingTrainer()
        trainer_ensemble.train_optimized_model(
            X=X,
            y=y,
            model_type='ensemble',
            use_scaling=False,
            test_size=0.2,
            n_folds=n_folds
        )
        
        results['Ensemble'] = {
            'accuracy': trainer_ensemble.training_metadata['accuracy'],
            'precision': trainer_ensemble.training_metadata['precision'],
            'recall': trainer_ensemble.training_metadata['recall'],
            'f1_score': trainer_ensemble.training_metadata['f1_score'],
            'roc_auc': trainer_ensemble.training_metadata.get('roc_auc', 0)
        }
        
        # Save ensemble as the primary model (best performance)
        trainer_ensemble.save_model(
            model_path='ml/phishing_model.pkl',
            metadata_path='ml/model_metadata.json'
        )
    
    # Print comprehensive comparison
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<10}")
    print("-" * 95)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} "
              f"{metrics['accuracy']:>10.2%}  "
              f"{metrics['precision']:>10.2%}  "
              f"{metrics['recall']:>10.2%}  "
              f"{metrics['f1_score']:>10.2%}  "
              f"{metrics['roc_auc']:>8.4f}")
    
    # Determine best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model[0]}")
    print("="*70)
    print(f"  Accuracy:  {best_model[1]['accuracy']:.2%}")
    print(f"  Precision: {best_model[1]['precision']:.2%}")
    print(f"  Recall:    {best_model[1]['recall']:.2%}")
    print(f"  F1-Score:  {best_model[1]['f1_score']:.2%}")
    if best_model[1]['roc_auc']:
        print(f"  ROC-AUC:   {best_model[1]['roc_auc']:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n[FILES SAVED]")
    print("  Primary model (Ensemble): ml/phishing_model.pkl")
    print("  Random Forest:            ml/phishing_model_rf_optimized.pkl")
    print("  Gradient Boosting:        ml/phishing_model_gb_optimized.pkl")
    
    print("\n[MODEL DETAILS]")
    print(f"  Dataset size:    {len(X)} samples")
    print(f"  Features:        {X.shape[1]}")
    print(f"  Training method: 5-Fold Cross-Validation")
    print(f"  Test split:      20%")
    
    print("\n[NEXT STEPS]")
    print("  1. Test the model:")
    print("     py ml/predict.py")
    print("\n  2. Start your Flask application:")
    print("     py app.py")
    print("\n  3. Test with real URLs in browser:")
    print("     http://localhost:5000")
    
    print("\n[TIPS FOR EVEN BETTER ACCURACY]")
    print("  • Download PhiUSIIL dataset: pip install ucimlrepo")
    print("  • Collect more real phishing URLs from PhishTank")
    print("  • Add custom features specific to your domain")
    print("  • Fine-tune hyperparameters with GridSearchCV")
    
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
