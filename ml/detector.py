"""
Phishing Detection Module
Modular ML-based phishing detection using scikit-learn
"""

import re
import urllib.parse
from typing import Dict, List

# Optional numpy import for ML features
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Create a dummy np for type checking
    class DummyNP:
        def array(self, *args, **kwargs):
            raise ImportError("numpy is required for ML model predictions")
    np = DummyNP()

class PhishingDetector:
    """
    Modular phishing detector with ML integration support
    Designed to be extended with scikit-learn models
    """
    
    def __init__(self):
        """Initialize detector with feature extractors"""
        self.suspicious_keywords = [
            'secure-login', 'verify-account', 'update-account',
            'suspended', 'verify', 'confirm', 'urgent', 'immediate',
            'click-here', 'limited-time', 'act-now'
        ]
        
        self.trusted_domains = [
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
            'paypal.com', 'ebay.com', 'facebook.com', 'twitter.com',
            'linkedin.com', 'github.com', 'stackoverflow.com'
        ]
    
    def extract_features(self, url: str) -> Dict[str, float]:
        """
        Extract features from URL for ML model
        Returns dictionary of features that can be used with scikit-learn
        """
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        full_url = url.lower()
        
        features = {
            'url_length': len(url),
            'domain_length': len(domain),
            'path_length': len(path),
            'num_dots': domain.count('.'),
            'num_hyphens': domain.count('-'),
            'num_slashes': url.count('/'),
            'num_question_marks': url.count('?'),
            'num_equals': url.count('='),
            'num_ampersands': url.count('&'),
            'num_percent': url.count('%'),
            'has_ip': 1.0 if self._is_ip_address(domain) else 0.0,
            'has_port': 1.0 if ':' in domain else 0.0,
            'suspicious_keyword_count': sum(1 for kw in self.suspicious_keywords if kw in full_url),
            'is_trusted_domain': 1.0 if any(td in domain for td in self.trusted_domains) else 0.0,
            'has_homograph': 1.0 if self._detect_homograph(domain) else 0.0,
            'subdomain_count': len([s for s in domain.split('.') if s]) - 2,  # Subtract domain and TLD
            'path_depth': len([p for p in path.split('/') if p]),
        }
        
        return features
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return bool(re.match(ip_pattern, domain))
    
    def _detect_homograph(self, domain: str) -> bool:
        """Detect homograph attacks (Cyrillic characters, etc.)"""
        # Check for non-ASCII characters that might be homographs
        for char in domain:
            if ord(char) > 127:
                return True
        return False
    
    def _calculate_threat_score(self, features: Dict[str, float]) -> float:
        """
        Calculate threat score based on features
        This is a rule-based approach that can be replaced with ML model
        """
        score = 0.0
        
        # URL length penalty
        if features['url_length'] > 100:
            score += 0.1
        
        # Suspicious keywords
        score += features['suspicious_keyword_count'] * 0.15
        
        # IP address usage
        if features['has_ip']:
            score += 0.3
        
        # Homograph detection
        if features['has_homograph']:
            score += 0.4
        
        # Too many subdomains
        if features['subdomain_count'] > 3:
            score += 0.1
        
        # Not a trusted domain
        if not features['is_trusted_domain']:
            score += 0.1
        
        return min(score, 1.0)
    
    def detect(self, url: str) -> Dict:
        """
        Main detection method
        Returns dictionary with detection results
        """
        try:
            features = self.extract_features(url)
            threat_score = self._calculate_threat_score(features)
            
            # Determine threat level
            if threat_score < 0.2:
                threat_level = 'safe'
                is_safe = True
            elif threat_score < 0.4:
                threat_level = 'low'
                is_safe = False
            elif threat_score < 0.6:
                threat_level = 'medium'
                is_safe = False
            elif threat_score < 0.8:
                threat_level = 'high'
                is_safe = False
            else:
                threat_level = 'critical'
                is_safe = False
            
            # Generate explanation
            explanation = self._generate_explanation(url, features, threat_score, threat_level)
            
            # Determine threat type
            threat_type = self._determine_threat_type(features, threat_score)
            
            return {
                'is_safe': is_safe,
                'threat_level': threat_level,
                'threat_type': threat_type,
                'explanation': explanation,
                'ml_score': round(threat_score, 3),
                'features': features
            }
            
        except Exception as e:
            # Fallback to safe if error occurs
            return {
                'is_safe': True,
                'threat_level': 'safe',
                'threat_type': None,
                'explanation': f'Unable to analyze URL: {str(e)}',
                'ml_score': 0.0
            }
    
    def _generate_explanation(self, url: str, features: Dict[str, float], 
                            threat_score: float, threat_level: str) -> str:
        """Generate human-readable explanation"""
        reasons = []
        
        if features['has_homograph']:
            reasons.append("This URL contains suspicious characters that may be used in homograph attacks")
        
        if features['has_ip']:
            reasons.append("The URL uses an IP address instead of a domain name")
        
        if features['suspicious_keyword_count'] > 0:
            reasons.append(f"Contains {int(features['suspicious_keyword_count'])} suspicious keywords")
        
        if features['subdomain_count'] > 3:
            reasons.append("Unusually high number of subdomains detected")
        
        if not features['is_trusted_domain'] and threat_score > 0.3:
            reasons.append("Domain does not match known trusted domains")
        
        if threat_level == 'safe':
            return "This site appears to be safe. No obvious phishing indicators detected."
        elif reasons:
            return f"Warning: {' '.join(reasons)}. Threat level: {threat_level.upper()}."
        else:
            return f"Threat level: {threat_level.upper()}. Exercise caution when visiting this site."
    
    def _determine_threat_type(self, features: Dict[str, float], threat_score: float) -> str:
        """Determine specific threat type"""
        if features['has_homograph']:
            return 'Homograph Attack'
        elif features['has_ip']:
            return 'IP Address Spoofing'
        elif features['suspicious_keyword_count'] > 2:
            return 'Credential Harvesting'
        elif threat_score > 0.6:
            return 'Suspicious Domain'
        elif threat_score > 0.4:
            return 'Domain Spoofing'
        else:
            return 'Low Risk Phishing'
    
    def predict_with_model(self, url: str, model) -> Dict:
        """
        Use scikit-learn model for prediction
        This method can be called when ML model is trained and loaded
        Requires numpy to be installed
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for ML model predictions. Install it with: pip install numpy")
        
        features = self.extract_features(url)
        
        # Convert features dict to numpy array in correct order
        feature_vector = np.array([
            features['url_length'],
            features['domain_length'],
            features['path_length'],
            features['num_dots'],
            features['num_hyphens'],
            features['num_slashes'],
            features['num_question_marks'],
            features['num_equals'],
            features['num_ampersands'],
            features['num_percent'],
            features['has_ip'],
            features['has_port'],
            features['suspicious_keyword_count'],
            features['is_trusted_domain'],
            features['has_homograph'],
            features['subdomain_count'],
            features['path_depth'],
        ]).reshape(1, -1)
        
        # Get prediction from model
        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0]
        
        # Map prediction to threat level
        threat_levels = ['safe', 'low', 'medium', 'high', 'critical']
        threat_level = threat_levels[min(int(prediction), len(threat_levels) - 1)]
        
        return {
            'is_safe': prediction == 0,
            'threat_level': threat_level,
            'ml_score': float(max(probability)),
            'features': features
        }
