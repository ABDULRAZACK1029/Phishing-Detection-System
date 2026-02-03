"""
Synthetic URL Evaluator
========================

This module evaluates phishing detection model performance on synthetic URLs.

ACADEMIC PURPOSE:
-----------------
This evaluator is designed for educational cybersecurity research to test
ML model robustness using ethically-generated synthetic phishing URLs.
All URLs use safe TLDs (.test, .example, .invalid) that will never be
registered as real domains.

Features:
- Automatic labeling of synthetic URLs
- Integration with existing FeatureExtractor pipeline
- Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
- Detailed performance reporting
- Export capabilities (JSON, CSV)

Author: BCA Final Year Project - Phishing Detection System
License: Educational Use Only
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import json
from datetime import datetime
import os

# Import existing modules
try:
    from ml.feature_extraction import FeatureExtractor
    from ml.predict import PhishingPredictor
except ImportError:
    from feature_extraction import FeatureExtractor
    from predict import PhishingPredictor


class SyntheticURLEvaluator:
    """
    Evaluator for phishing detection models using synthetic URLs.
    
    This class provides comprehensive evaluation capabilities including:
    - Feature extraction from synthetic URLs
    - Model prediction and comparison with ground truth
    - Metrics calculation (accuracy, precision, recall, F1)
    - Confusion matrix generation
    - Detailed reporting and export
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained model (uses default if None)
        """
        self.feature_extractor = FeatureExtractor()
        self.predictor = PhishingPredictor(model_path) if model_path else PhishingPredictor()
        self.results = {}
    
    def evaluate_urls(
        self,
        urls: List[str],
        labels: List[int],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on synthetic URLs.
        
        Args:
            urls: List of synthetic URLs to evaluate
            labels: Ground truth labels (0=legitimate, 1=phishing)
            verbose: Print detailed progress
        
        Returns:
            Dictionary containing evaluation metrics and results
        """
        if len(urls) != len(labels):
            raise ValueError("URLs and labels must have the same length")
        
        if verbose:
            print("=" * 70)
            print("SYNTHETIC URL EVALUATION")
            print("=" * 70)
            print(f"\nEvaluating {len(urls)} synthetic URLs...")
            print(f"  Phishing URLs: {sum(labels)}")
            print(f"  Legitimate URLs: {len(labels) - sum(labels)}")
        
        # Extract features and make predictions
        predictions = []
        prediction_probas = []
        feature_vectors = []
        
        if verbose:
            print("\n[*] Extracting features and making predictions...")
        
        for i, url in enumerate(urls):
            if verbose and (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(urls)} URLs...")
            
            # Get prediction
            result = self.predictor.predict_url(url)
            
            # Store results
            pred_label = 0 if result['is_safe'] else 1
            predictions.append(pred_label)
            prediction_probas.append(result['confidence'])
            feature_vectors.append(result['features'])
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Calculate metrics
        if verbose:
            print("\n[*] Calculating metrics...")
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Try to calculate ROC-AUC
        try:
            # Convert probabilities based on predictions
            proba_for_roc = []
            for i, pred in enumerate(predictions):
                if pred == 1:  # Phishing
                    proba_for_roc.append(prediction_probas[i])
                else:  # Legitimate
                    proba_for_roc.append(1 - prediction_probas[i])
            roc_auc = roc_auc_score(labels, proba_for_roc)
        except Exception as e:
            roc_auc = None
            if verbose:
                print(f"    Warning: Could not calculate ROC-AUC: {e}")
        
        # Store results
        self.results = {
            'evaluation_date': datetime.now().isoformat(),
            'total_urls': len(urls),
            'phishing_urls': int(sum(labels)),
            'legitimate_urls': int(len(labels) - sum(labels)),
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'false_positive_rate': float(false_positive_rate),
                'false_negative_rate': float(false_negative_rate),
                'roc_auc': float(roc_auc) if roc_auc else None
            },
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'predictions': predictions.tolist(),
            'ground_truth': labels.tolist(),
            'urls': urls,
            'feature_vectors': feature_vectors
        }
        
        # Print results
        if verbose:
            self.print_results()
        
        return self.results
    
    def print_results(self):
        """Print formatted evaluation results."""
        if not self.results:
            print("No results to display. Run evaluate_urls() first.")
            return
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        metrics = self.results['metrics']
        cm = self.results['confusion_matrix']
        
        print(f"\n[*] Overall Performance:")
        print(f"    Accuracy:  {metrics['accuracy']:.2%}")
        print(f"    Precision: {metrics['precision']:.2%}")
        print(f"    Recall:    {metrics['recall']:.2%}")
        print(f"    F1-Score:  {metrics['f1_score']:.2%}")
        
        if metrics['roc_auc']:
            print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\n[*] Additional Metrics:")
        print(f"    Specificity (True Negative Rate): {metrics['specificity']:.2%}")
        print(f"    False Positive Rate: {metrics['false_positive_rate']:.2%}")
        print(f"    False Negative Rate: {metrics['false_negative_rate']:.2%}")
        
        print(f"\n[*] Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                 Legit  Phishing")
        print(f"    Actual Legit   {cm['true_negatives']:5d}    {cm['false_positives']:5d}")
        print(f"           Phish   {cm['false_negatives']:5d}    {cm['true_positives']:5d}")
        
        print(f"\n[*] Interpretation:")
        print(f"    True Positives (TP):  {cm['true_positives']} - Correctly identified phishing URLs")
        print(f"    True Negatives (TN):  {cm['true_negatives']} - Correctly identified legitimate URLs")
        print(f"    False Positives (FP): {cm['false_positives']} - Legitimate URLs flagged as phishing")
        print(f"    False Negatives (FN): {cm['false_negatives']} - Phishing URLs missed by model")
        
        # Performance assessment
        print(f"\n[*] Performance Assessment:")
        if metrics['accuracy'] >= 0.95:
            print("    ✓ EXCELLENT - Model performs very well on synthetic URLs")
        elif metrics['accuracy'] >= 0.85:
            print("    ✓ GOOD - Model performs well with room for improvement")
        elif metrics['accuracy'] >= 0.75:
            print("    ⚠ MODERATE - Model needs improvement")
        else:
            print("    ✗ POOR - Model struggles with synthetic phishing patterns")
        
        if metrics['false_positive_rate'] > 0.1:
            print(f"    ⚠ WARNING: High false positive rate ({metrics['false_positive_rate']:.1%})")
            print("       Many legitimate URLs are incorrectly flagged as phishing")
        
        if metrics['false_negative_rate'] > 0.1:
            print(f"    ⚠ WARNING: High false negative rate ({metrics['false_negative_rate']:.1%})")
            print("       Many phishing URLs are missed by the model")
        
        print("=" * 70)
    
    def export_results(
        self,
        filepath: str,
        format: str = 'json',
        include_urls: bool = True
    ):
        """
        Export evaluation results to file.
        
        Args:
            filepath: Output file path
            format: 'json' or 'csv'
            include_urls: Include full URL list in export
        """
        if not self.results:
            raise ValueError("No results to export. Run evaluate_urls() first.")
        
        # Prepare export data
        export_data = self.results.copy()
        
        if not include_urls:
            export_data.pop('urls', None)
            export_data.pop('feature_vectors', None)
            export_data.pop('predictions', None)
            export_data.pop('ground_truth', None)
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"[OK] Results exported to: {filepath}")
        
        elif format == 'csv':
            # Create summary CSV
            metrics = export_data['metrics']
            cm = export_data['confusion_matrix']
            
            summary_data = {
                'Metric': [
                    'Accuracy', 'Precision', 'Recall', 'F1-Score',
                    'Specificity', 'False Positive Rate', 'False Negative Rate',
                    'True Positives', 'True Negatives', 'False Positives', 'False Negatives'
                ],
                'Value': [
                    metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'],
                    metrics['specificity'], metrics['false_positive_rate'], metrics['false_negative_rate'],
                    cm['true_positives'], cm['true_negatives'], cm['false_positives'], cm['false_negatives']
                ]
            }
            
            df = pd.DataFrame(summary_data)
            df.to_csv(filepath, index=False)
            print(f"[OK] Results exported to: {filepath}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")
    
    def compare_with_baseline(
        self,
        baseline_metrics: Dict[str, float]
    ):
        """
        Compare current results with baseline metrics.
        
        Args:
            baseline_metrics: Dictionary with baseline accuracy, precision, recall, f1
        """
        if not self.results:
            raise ValueError("No results to compare. Run evaluate_urls() first.")
        
        current = self.results['metrics']
        
        print("\n" + "=" * 70)
        print("COMPARISON WITH BASELINE")
        print("=" * 70)
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics_to_compare:
            if metric in baseline_metrics and metric in current:
                baseline_val = baseline_metrics[metric]
                current_val = current[metric]
                diff = current_val - baseline_val
                
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  Baseline: {baseline_val:.2%}")
                print(f"  Current:  {current_val:.2%}")
                print(f"  Change:   {symbol} {abs(diff):.2%}")
        
        print("=" * 70)
    
    def get_misclassified_urls(self) -> Dict[str, List[str]]:
        """
        Get URLs that were misclassified.
        
        Returns:
            Dictionary with 'false_positives' and 'false_negatives' lists
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate_urls() first.")
        
        urls = self.results['urls']
        predictions = self.results['predictions']
        ground_truth = self.results['ground_truth']
        
        false_positives = []
        false_negatives = []
        
        for i, (url, pred, true) in enumerate(zip(urls, predictions, ground_truth)):
            if pred == 1 and true == 0:
                false_positives.append(url)
            elif pred == 0 and true == 1:
                false_negatives.append(url)
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def analyze_errors(self, max_display: int = 10):
        """
        Analyze and display misclassified URLs.
        
        Args:
            max_display: Maximum number of errors to display per category
        """
        misclassified = self.get_misclassified_urls()
        
        print("\n" + "=" * 70)
        print("ERROR ANALYSIS")
        print("=" * 70)
        
        print(f"\n[*] False Positives (Legitimate flagged as Phishing): {len(misclassified['false_positives'])}")
        for i, url in enumerate(misclassified['false_positives'][:max_display], 1):
            print(f"    {i}. {url}")
        
        if len(misclassified['false_positives']) > max_display:
            print(f"    ... and {len(misclassified['false_positives']) - max_display} more")
        
        print(f"\n[*] False Negatives (Phishing missed): {len(misclassified['false_negatives'])}")
        for i, url in enumerate(misclassified['false_negatives'][:max_display], 1):
            print(f"    {i}. {url}")
        
        if len(misclassified['false_negatives']) > max_display:
            print(f"    ... and {len(misclassified['false_negatives']) - max_display} more")
        
        print("=" * 70)


# Convenience function
def evaluate_synthetic_urls(
    urls: List[str],
    labels: List[int],
    model_path: Optional[str] = None,
    export_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate synthetic URLs.
    
    Args:
        urls: List of synthetic URLs
        labels: Ground truth labels (0=legitimate, 1=phishing)
        model_path: Path to model (optional)
        export_path: Path to export results (optional)
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = SyntheticURLEvaluator(model_path)
    results = evaluator.evaluate_urls(urls, labels)
    
    if export_path:
        evaluator.export_results(export_path)
    
    return results


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("SYNTHETIC URL EVALUATOR - DEMO")
    print("=" * 70)
    print("\nThis demo requires a trained model and synthetic URLs.")
    print("Run synthetic_url_generator.py first to generate test URLs.\n")
    
    # Example with dummy data
    from synthetic_url_generator import generate_synthetic_dataset
    
    print("[*] Generating 100 synthetic URLs for testing...")
    urls, labels = generate_synthetic_dataset(count=100, realism="medium")
    
    print(f"[OK] Generated {len(urls)} URLs")
    print(f"    Phishing: {sum(labels)}")
    print(f"    Legitimate: {len(labels) - sum(labels)}")
    
    print("\n[*] Evaluating model performance...")
    evaluator = SyntheticURLEvaluator()
    
    try:
        results = evaluator.evaluate_urls(urls, labels)
        
        # Show error analysis
        evaluator.analyze_errors(max_display=5)
        
        # Export results
        export_path = "ml/synthetic_evaluation_results.json"
        evaluator.export_results(export_path, format='json', include_urls=False)
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        print("Make sure you have a trained model at ml/phishing_model.pkl")
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)
