"""
Prediction Module for Phishing Detection
Loads trained ML model and provides prediction API

This module provides a clean interface for making predictions on URLs
using the trained Random Forest or Logistic Regression model.
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Optional
import threading

# Import feature extractor
try:
    from ml.feature_extraction import FeatureExtractor
except ImportError:
    from feature_extraction import FeatureExtractor


class PhishingPredictor:
    """
    Phishing URL predictor using trained ML model.
    
    This class provides thread-safe model loading and prediction capabilities.
    The model is lazily loaded on first prediction to improve startup time.
    
    Usage:
        predictor = PhishingPredictor()
        result = predictor.predict_url('https://example.com')
        print(result['prediction'])  # 'Phishing' or 'Legitimate'
        print(result['confidence'])  # 0.0 to 1.0
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, model_path: str = 'ml/phishing_model.pkl', 
                 metadata_path: str = 'ml/model_metadata.json'):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the trained model file
            metadata_path (str): Path to model metadata JSON
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.feature_extractor = FeatureExtractor()
        self._model_loaded = False
    
    def _load_model(self) -> bool:
        """
        Load the trained model from disk (lazy loading).
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True
        
        with self._lock:
            # Double-check pattern for thread safety
            if self._model_loaded:
                return True
            
            try:
                # Check if model file exists
                if not os.path.exists(self.model_path):
                    print(f"⚠️  Model file not found: {self.model_path}")
                    print("   Run 'python ml/trainer.py' to train the model first.")
                    return False
                
                # Load the trained model
                self.model = joblib.load(self.model_path)
                print(f"[OK] ML model loaded from: {self.model_path}")
                
                # Load metadata if available
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    print(f"[OK] Model metadata loaded (Accuracy: {self.metadata.get('accuracy', 0):.2%})")
                
                self._model_loaded = True
                return True
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return False
    
    def predict_url(self, url: str) -> Dict:
        """
        Predict whether a URL is phishing or legitimate.
        
        Args:
            url (str): The URL to analyze
            
        Returns:
            Dict containing:
                - prediction (str): 'Phishing' or 'Legitimate'
                - confidence (float): Confidence score 0.0 to 1.0
                - is_safe (bool): True if legitimate, False if phishing
                - threat_level (str): 'safe', 'low', 'medium', 'high', 'critical'
                - ml_score (float): Raw ML confidence score
                - features (dict): Extracted features from URL
                - model_available (bool): Whether ML model was used
        """
        # Extract features from URL
        features = self.feature_extractor.extract_features(url)
        
        # Try to load model if not loaded
        if not self._model_loaded:
            if not self._load_model():
                # Fallback to rule-based detection
                return self._fallback_prediction(url, features)
        
        try:
            # Get feature vector in correct order
            feature_vector = self.feature_extractor.get_feature_vector(url).reshape(1, -1)
            
            # Make prediction
            prediction_label = self.model.predict(feature_vector)[0]
            prediction_proba = self.model.predict_proba(feature_vector)[0]
            
            # Get confidence score for the predicted class
            # For binary classification: proba[0] = legitimate, proba[1] = phishing
            if prediction_label == 1:
                # Phishing - use phishing probability
                confidence = float(prediction_proba[1])
                prediction_text = 'Phishing'
                is_safe = False
            else:
                # Legitimate - use legitimate probability
                confidence = float(prediction_proba[0])
                prediction_text = 'Legitimate'
                is_safe = True
            
            # Map confidence to threat level
            threat_level = self._confidence_to_threat_level(confidence, is_safe)
            
            # Prepare result
            result = {
                'prediction': prediction_text,
                'confidence': round(confidence, 4),
                'is_safe': is_safe,
                'threat_level': threat_level,
                'ml_score': round(confidence if not is_safe else (1.0 - confidence), 4),
                'features': features,
                'model_available': True
            }
            
            return result
            
        except Exception as e:
            print(f"[X] Prediction error: {e}")
            # Fallback to rule-based detection
            return self._fallback_prediction(url, features)
    
    def _confidence_to_threat_level(self, confidence: float, is_safe: bool) -> str:
        """
        Map ML confidence score to threat level category.
        
        Args:
            confidence (float): Model confidence score (0.0 to 1.0)
            is_safe (bool): Whether URL is predicted as safe
            
        Returns:
            str: Threat level ('safe', 'low', 'medium', 'high', 'critical')
        """
        if is_safe:
            # For safe URLs, high confidence = safe
            if confidence > 0.9:
                return 'safe'
            elif confidence > 0.7:
                return 'low'
            else:
                return 'medium'
        else:
            # For phishing URLs, map confidence to severity
            if confidence > 0.9:
                return 'critical'
            elif confidence > 0.75:
                return 'high'
            elif confidence > 0.6:
                return 'medium'
            else:
                return 'low'
    
    def _fallback_prediction(self, url: str, features: Dict) -> Dict:
        """
        Fallback to rule-based prediction when ML model is unavailable.
        
        Args:
            url (str): The URL being analyzed
            features (Dict): Extracted features
            
        Returns:
            Dict: Prediction result using rule-based approach
        """
        # Simple rule-based scoring
        threat_score = 0.0
        
        # URL length penalty
        if features['url_length'] > 100:
            threat_score += 0.1
        
        # Suspicious keywords
        threat_score += min(features['suspicious_keyword_count'] * 0.15, 0.4)
        
        # IP address usage
        if features['has_ip']:
            threat_score += 0.3
        
        # Homograph detection
        if features['has_homograph']:
            threat_score += 0.4
        
        # Too many subdomains
        if features['subdomain_count'] > 3:
            threat_score += 0.1
        
        # Not a trusted domain
        if not features['is_trusted_domain']:
            threat_score += 0.05
        
        # Cap at 1.0
        threat_score = min(threat_score, 1.0)
        
        # Determine if safe
        is_safe = threat_score < 0.3
        
        # Map to threat level
        if threat_score < 0.2:
            threat_level = 'safe'
        elif threat_score < 0.4:
            threat_level = 'low'
        elif threat_score < 0.6:
            threat_level = 'medium'
        elif threat_score < 0.8:
            threat_level = 'high'
        else:
            threat_level = 'critical'
        
        return {
            'prediction': 'Legitimate' if is_safe else 'Phishing',
            'confidence': round(1.0 - threat_score if is_safe else threat_score, 4),
            'is_safe': is_safe,
            'threat_level': threat_level,
            'ml_score': round(threat_score, 4),
            'features': features,
            'model_available': False
        }


# Global predictor instance (singleton pattern)
_global_predictor: Optional[PhishingPredictor] = None


def get_predictor() -> PhishingPredictor:
    """
    Get or create the global predictor instance (singleton).
    
    Returns:
        PhishingPredictor: The global predictor instance
    """
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = PhishingPredictor()
    return _global_predictor


def predict_url(url: str) -> Dict:
    """
    Convenience function to predict a URL using the global predictor.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        Dict: Prediction result
        
    Example:
        >>> result = predict_url('https://secure-bank-verify.com/login')
        >>> print(f"Prediction: {result['prediction']}")
        >>> print(f"Confidence: {result['confidence']:.2%}")
    """
    predictor = get_predictor()
    return predictor.predict_url(url)


if __name__ == '__main__':
    # Test the predictor
    print("Phishing URL Predictor - Test\n")
    
    test_urls = [
        'https://www.google.com',
        'http://192.168.1.1/verify-account',
        'https://secure-login-update-bank.suspicious.com/verify?user=123&pass=abc',
        'https://github.com/user/repo',
        'http://paypal-secure-verify.phishing.com/login'
    ]
    
    for url in test_urls:
        print(f"\n{'='*70}")
        print(f"URL: {url}")
        print('-'*70)
        
        result = predict_url(url)
        
        print(f"Prediction:    {result['prediction']}")
        print(f"Confidence:    {result['confidence']:.2%}")
        print(f"Threat Level:  {result['threat_level'].upper()}")
        print(f"Is Safe:       {result['is_safe']}")
        print(f"ML Model Used: {'[OK]' if result['model_available'] else '[X] (using rule-based)'}")
        
        # Show key features
        features = result['features']
        print(f"\nKey Features:")
        print(f"  - URL Length: {features['url_length']}")
        print(f"  - Suspicious Keywords: {features['suspicious_keyword_count']}")
        print(f"  - Has IP: {'Yes' if features['has_ip'] else 'No'}")
        print(f"  - Trusted Domain: {'Yes' if features['is_trusted_domain'] else 'No'}")

