"""
Feature Extraction Module for Phishing Detection
Extracts URL-based features for ML model prediction

This module provides a clean, academic-grade implementation of feature extraction
suitable for BCA Final Year project viva defense and explanation.
"""

import re
import urllib.parse
from typing import Dict, List, Tuple
import numpy as np


class FeatureExtractor:
    """
    Extract features from URLs for phishing detection.
    
    This class implements 17 URL-based features that are commonly used in
    academic phishing detection research. Each feature is designed to capture
    different aspects of potentially malicious URLs.
    
    Features extracted:
    1. URL length - Phishing URLs are often longer to hide malicious intent
    2. Domain length - Suspicious domains may have unusual lengths
    3. Path length - Long paths may indicate parameter stuffing
    4. Number of dots - Multiple subdomains are suspicious
    5. Number of hyphens - Excessive hyphens may indicate spoofing
    6. Number of slashes - Path complexity indicator
    7. Number of question marks - Query parameter presence
    8. Number of equals signs - Query parameter count
    9. Number of ampersands - Multiple parameters indicator
    10. Number of percent signs - URL encoding usage
    11. Has IP address - Using IP instead of domain is suspicious
    12. Has port number - Non-standard ports may be malicious
    13. Suspicious keyword count - Common phishing terms
    14. Is trusted domain - Whitelist of known safe domains
    15. Has homograph characters - Unicode spoofing detection
    16. Subdomain count - Multiple subdomains can hide intent
    17. Path depth - Deep nesting may indicate hiding
    """
    
    def __init__(self):
        """Initialize feature extractor with configuration data."""
        # Suspicious keywords commonly found in phishing URLs
        self.suspicious_keywords = [
            'login', 'signin', 'verify', 'verification', 'confirm', 'confirmation',
            'update', 'secure', 'account', 'banking', 'suspend', 'suspended',
            'urgent', 'immediate', 'alert', 'warning', 'locked', 'unlock',
            'verify-account', 'update-account', 'secure-login', 'click-here',
            'limited-time', 'act-now', 'password', 'credential', 'security',
            'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
            'bank', 'credit', 'card', 'payment', 'billing'
        ]
        
        # Trusted domains (whitelist of known legitimate sites)
        self.trusted_domains = [
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
            'paypal.com', 'ebay.com', 'facebook.com', 'twitter.com',
            'instagram.com', 'linkedin.com', 'github.com', 'stackoverflow.com',
            'youtube.com', 'netflix.com', 'adobe.com', 'oracle.com',
            'ibm.com', 'salesforce.com', 'zoom.us', 'dropbox.com'
        ]
        
        
        
        # Feature names - Base features
        self.feature_names = [
            'url_length', 'domain_length', 'path_length', 'num_dots',
            'num_hyphens', 'num_slashes', 'num_question_marks', 'num_equals',
            'num_ampersands', 'num_percent', 'has_ip', 'has_port',
            'suspicious_keyword_count', 'is_trusted_domain', 'has_homograph',
            'subdomain_count', 'path_depth'
        ]
        
        # New advanced features
        self.feature_names.extend([
            'entropy', 'digit_ratio', 'longest_token_len', 'tld_in_path',
            'is_shortened', 'is_suspicious_tld', 'has_client_server',
            'domain_token_count', 'path_token_count', 'is_punycode'
        ])
        
        # Common URL shorteners
        self.shorteners = [
            'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd',
            'buff.ly', 'adf.ly', 'bit.do', 'tr.im'
        ]
        
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = [
            '.xyz', '.top', '.loan', '.click', '.country', '.stream',
            '.gdn', '.mom', '.win', '.review', '.vip', '.party'
        ]

    def extract_features(self, url: str) -> Dict[str, float]:
        """
        Extract all features from a URL.
        
        Args:
            url (str): The URL to analyze
            
        Returns:
            Dict[str, float]: Dictionary mapping feature names to their values
        """
        # Parse URL into components
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower() if parsed.netloc else ''
            path = parsed.path.lower() if parsed.path else ''
            full_url = url.lower()
        except Exception:
            # If URL parsing fails, return zero features
            return {name: 0.0 for name in self.feature_names}
        
        # Extract each feature
        features = {
            # Length-based features
            'url_length': float(len(url)),
            'domain_length': float(len(domain)),
            'path_length': float(len(path)),
            
            # Character count features
            'num_dots': float(domain.count('.')),
            'num_hyphens': float(domain.count('-')),
            'num_slashes': float(url.count('/')),
            'num_question_marks': float(url.count('?')),
            'num_equals': float(url.count('=')),
            'num_ampersands': float(url.count('&')),
            'num_percent': float(url.count('%')),
            
            # Binary features (0 or 1)
            'has_ip': 1.0 if self._is_ip_address(domain) else 0.0,
            'has_port': 1.0 if ':' in domain.replace('http://', '').replace('https://', '') else 0.0,
            'is_trusted_domain': 1.0 if self._is_trusted_domain(domain) else 0.0,
            'has_homograph': 1.0 if self._has_homograph_chars(domain) else 0.0,
            
            # Count-based features
            'suspicious_keyword_count': float(self._count_suspicious_keywords(full_url)),
            'subdomain_count': float(self._count_subdomains(domain)),
            'path_depth': float(self._calculate_path_depth(path)),
            
            # --- NEW FEATURES ---
            'entropy': self._calculate_entropy(domain),
            'digit_ratio': self._calculate_digit_ratio(url),
            'longest_token_len': float(self._longest_token_length(url)),
            'tld_in_path': 1.0 if self._tld_in_path(path) else 0.0,
            'is_shortened': 1.0 if self._is_shortened(domain) else 0.0,
            'is_suspicious_tld': 1.0 if self._is_suspicious_tld(domain) else 0.0,
            'has_client_server': 1.0 if 'client' in full_url or 'server' in full_url else 0.0,
            'domain_token_count': float(len(re.split(r'[.-]', domain))),
            'path_token_count': float(len(re.split(r'[/-]', path)) if path else 0),
            'is_punycode': 1.0 if 'xn--' in domain else 0.0,
        }
        
        return features
    
    def get_feature_vector(self, url: str) -> np.ndarray:
        """
        Extract features and return as ordered numpy array for ML model.
        
        Args:
            url (str): The URL to analyze
            
        Returns:
            np.ndarray: Feature vector (1D array) in the correct order for the model
        """
        features = self.extract_features(url)
        # Return features in the specific order required by the model
        return np.array([features[name] for name in self.feature_names])
    
    def get_feature_names(self) -> List[str]:
        """
        Get the ordered list of feature names.
        
        Returns:
            List[str]: Feature names in the order used by the model
        """
        return self.feature_names.copy()
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if the domain is an IP address."""
        # IPv4 pattern: xxx.xxx.xxx.xxx
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        domain_without_port = domain.split(':')[0]
        
        if re.match(ipv4_pattern, domain_without_port):
            octets = domain_without_port.split('.')
            return all(0 <= int(octet) <= 255 for octet in octets)
        
        if domain_without_port.count(':') >= 2:
            return True
        return False
    
    def _is_trusted_domain(self, domain: str) -> bool:
        """Check if the domain is in the trusted domain whitelist or has a trusted TLD."""
        domain_clean = domain.split(':')[0].lower()
        
        # 1. Check Trusted TLDs (Education/Government are rarely used for phishing)
        trusted_tlds = ['.edu', '.gov', '.mil', '.ac.in', '.edu.in', '.gov.in', '.org', '.int']
        if any(domain_clean.endswith(tld) for tld in trusted_tlds):
            return True
            
        # 2. Check Whitelisted Domains
        for trusted in self.trusted_domains:
            if trusted in domain_clean:
                if domain_clean.endswith(trusted) or domain_clean == trusted:
                    return True
        return False
    
    def _has_homograph_chars(self, domain: str) -> bool:
        """Detect potential homograph attacks using non-ASCII characters."""
        for char in domain:
            if ord(char) > 127:
                return True
        return False
    
    def _count_suspicious_keywords(self, url: str) -> int:
        """Count how many suspicious keywords appear in the URL."""
        count = 0
        for keyword in self.suspicious_keywords:
            if keyword in url:
                count += 1
        return count
    
    def _count_subdomains(self, domain: str) -> int:
        """Count the number of subdomains in the domain."""
        if not domain or self._is_ip_address(domain):
            return 0
        domain_clean = domain.split(':')[0]
        parts = [p for p in domain_clean.split('.') if p]
        return max(len(parts) - 2, 0)
    
    def _calculate_path_depth(self, path: str) -> int:
        """Calculate the depth of the URL path."""
        if not path or path == '/':
            return 0
        parts = [p for p in path.split('/') if p]
        return len(parts)

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the string."""
        if not text:
            return 0.0
        text = text.lower()
        probs = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * np.log2(p) for p in probs)

    def _calculate_digit_ratio(self, text: str) -> float:
        """Calculate the ratio of digits to total characters."""
        if not text:
            return 0.0
        digits = sum(c.isdigit() for c in text)
        return digits / len(text)

    def _longest_token_length(self, text: str) -> int:
        """Find the length of the longest alphanumeric token."""
        tokens = re.split(r'[^a-zA-Z0-9]', text)
        if not tokens:
            return 0
        return len(max(tokens, key=len))

    def _tld_in_path(self, path: str) -> bool:
        """Check if a common TLD appears in the path (e.g., /com/)."""
        common_tlds = ['.com', '.net', '.org', '.edu', '.gov']
        return any(tld in path for tld in common_tlds)

    def _is_shortened(self, domain: str) -> bool:
        """Check if the domain is a known URL shortener."""
        domain_clean = domain.split(':')[0]
        return any(short in domain_clean for short in self.shorteners)

    def _is_suspicious_tld(self, domain: str) -> bool:
        """Check if the TLD is considered suspicious."""
        return any(domain.endswith(tld) for tld in self.suspicious_tlds)


# Convenience function for standalone use
def extract_url_features(url: str) -> Dict[str, float]:
    """
    Convenience function to extract features from a URL.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        Dict[str, float]: Dictionary of features
    """
    extractor = FeatureExtractor()
    return extractor.extract_features(url)


if __name__ == '__main__':
    # Example usage for testing
    print("Feature Extraction Module - Test\n")
    
    extractor = FeatureExtractor()
    
    # Test URLs
    test_urls = [
        'https://www.google.com',
        'http://192.168.1.1/verify-account',
        'https://secure-login-verify.suspicious-bank.com/update/password?user=123',
        'https://www.github.com/user/repository',
        'http://bit.ly/suspicious',
        'http://cheap-loan.xyz/login'
    ]
    
    for url in test_urls:
        print(f"\nURL: {url}")
        features = extractor.extract_features(url)
        print(f"  Entropy: {features['entropy']:.4f}")
        print(f"  Digit Ratio: {features['digit_ratio']:.4f}")
        print(f"  Shortened: {features['is_shortened']}")
        print(f"  Suspicious TLD: {features['is_suspicious_tld']}")

