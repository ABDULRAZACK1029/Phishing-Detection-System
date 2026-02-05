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
            'domain_token_count', 'path_token_count', 'is_punycode',
            'has_ascii_homoglyph'  # NEW: ASCII homoglyph detection
        ])
        
        # Common URL shorteners (expanded database)
        self.shorteners = [
            # Original popular shorteners
            'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd',
            'buff.ly', 'adf.ly', 'bit.do', 'tr.im',
            # Additional shorteners
            'short.io', 'rebrand.ly', 'cutt.ly', 'tiny.cc', 's.id',
            'clck.ru', 'shorturl.at', 'tinycc.com', 'v.gd', 'x.co',
            'ead.me', 'lnkd.in', 'soo.gd', 'ity.im', 'q.gs',
            'po.st', 'bc.vc', 'u.to', 'j.mp', 'tweez.me',
            'shorte.st', 'mcaf.ee', 'su.pr', 'filoops.info', 'linktr.ee'
        ]
        
        # Shortener TLDs (commonly used for URL shortening)
        self.shortener_tlds = ['.ly', '.co', '.me', '.io', '.gl', '.gd']
        
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = [
            '.xyz', '.top', '.loan', '.click', '.country', '.stream',
            '.gdn', '.mom', '.win', '.review', '.vip', '.party'
        ]
        
        # Brand names commonly targeted in phishing attacks
        self.brand_names = [
            'amazon', 'paypal', 'facebook', 'google', 'microsoft', 'apple',
            'instagram', 'twitter', 'linkedin', 'netflix', 'ebay', 'yahoo',
            'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'usbank',
            'americanexpress', 'discover', 'capitalone', 'hsbc', 'barclays',
            'whatsapp', 'telegram', 'snapchat', 'tiktok', 'reddit',
            'dropbox', 'adobe', 'salesforce', 'oracle', 'ibm', 'zoom'
        ]
        
        # Homoglyph character mappings (ASCII lookalikes)
        # Maps suspicious characters to their legitimate equivalents
        self.homoglyph_map = {
            '0': 'o',  # zero to letter o
            '1': 'l',  # one to letter l
            '3': 'e',  # three to letter e
            '5': 's',  # five to letter s
            '8': 'b',  # eight to letter b
            '9': 'g',  # nine to letter g
        }
        
        # Multi-character homoglyphs (character sequences that look like single chars)
        self.multi_char_homoglyphs = [
            ('rn', 'm'),   # rn looks like m
            ('vv', 'w'),   # vv looks like w
            ('cl', 'd'),   # cl looks like d
            ('nn', 'u'),   # nn can look like u
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
            'has_ascii_homoglyph': 1.0 if self._detect_ascii_homoglyph(domain) else 0.0,
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
        """Check if the domain is a known URL shortener using database and patterns."""
        domain_clean = domain.split(':')[0].lower()
        
        # Check against known shortener database
        if any(short in domain_clean for short in self.shorteners):
            return True
        
        # Pattern-based detection (for unknown shorteners)
        return self._is_shortener_pattern(domain_clean)
    
    def _is_shortener_pattern(self, domain: str) -> bool:
        """
        Detect URL shortener patterns using heuristics.
        
        Patterns detected:
        1. Single-letter subdomain (l.ead.me, t.co, s.id)
        2. Very short domain (≤8 chars) with shortener TLD
        3. Short domain with common shortener patterns
        
        Args:
            domain (str): Domain to check
            
        Returns:
            bool: True if domain matches shortener patterns
        """
        if not domain:
            return False
        
        # Remove port if present
        domain = domain.split(':')[0]
        
        # Pattern 1: Single-letter subdomain
        # Examples: l.ead.me, t.co, s.id, x.co
        parts = domain.split('.')
        if len(parts) >= 2:
            subdomain = parts[0]
            if len(subdomain) == 1 and subdomain.isalpha():
                # Single letter subdomain is very common for shorteners
                return True
        
        # Pattern 2: Very short domain with shortener TLD
        # Examples: bit.ly, goo.gl, t.co
        if len(domain) <= 8:
            for tld in self.shortener_tlds:
                if domain.endswith(tld):
                    return True
        
        # Pattern 3: Two-letter domain with common TLDs
        # Examples: t.co, x.co, v.gd
        if len(parts) == 2:
            main_domain = parts[0]
            if len(main_domain) <= 2 and main_domain.isalpha():
                return True
        
        return False

    def _is_suspicious_tld(self, domain: str) -> bool:
        """Check if the TLD is considered suspicious."""
        return any(domain.endswith(tld) for tld in self.suspicious_tlds)
    
    def _normalize_homoglyphs(self, text: str) -> str:
        """
        Normalize a string by replacing homoglyph characters with their legitimate equivalents.
        
        This helps detect domains like 'amaz0n.com' (with zero) vs 'amazon.com' (with letter o).
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text with homoglyphs replaced
        """
        normalized = text.lower()
        
        # Replace single-character homoglyphs
        for suspicious, legitimate in self.homoglyph_map.items():
            normalized = normalized.replace(suspicious, legitimate)
        
        # Replace multi-character homoglyphs
        for suspicious_seq, legitimate_char in self.multi_char_homoglyphs:
            normalized = normalized.replace(suspicious_seq, legitimate_char)
        
        return normalized
    
    def _detect_ascii_homoglyph(self, domain: str) -> bool:
        """
        Detect ASCII-based homoglyph attacks in domain names.
        
        This catches phishing attempts like:
        - amazorn.com (rn -> m)
        - faceb00k.com (0 -> o)
        - paypa1.com (1 -> l)
        - g00gle.com (0 -> o)
        
        Args:
            domain (str): Domain name to check
            
        Returns:
            bool: True if homoglyph attack detected, False otherwise
        """
        if not domain:
            return False
        
        # Remove port and clean domain
        domain_clean = domain.split(':')[0].lower()
        
        # Remove common TLD to focus on brand name
        # e.g., 'faceb00k.com' -> 'faceb00k'
        for tld in ['.com', '.net', '.org', '.co', '.io', '.edu', '.gov']:
            if domain_clean.endswith(tld):
                domain_clean = domain_clean[:-len(tld)]
                break
        
        # Also remove subdomains - focus on main domain
        # e.g., 'login.faceb00k' -> 'faceb00k'
        parts = domain_clean.split('.')
        if len(parts) > 1:
            domain_clean = parts[-1]  # Get the last part (main domain)
        
        # Normalize the domain by replacing homoglyphs
        normalized = self._normalize_homoglyphs(domain_clean)
        
        # Check if normalized domain matches any known brand
        for brand in self.brand_names:
            # Exact match after normalization
            if normalized == brand:
                # The normalized domain matches a brand, but does the original?
                # If original doesn't match, it's a homoglyph attack
                if domain_clean != brand:
                    return True
            
            # Check if brand is contained in normalized domain
            # e.g., 'faceb00k-login' -> 'facebook-login' (normalized)
            if brand in normalized and brand not in domain_clean:
                # Brand appears after normalization but not in original
                return True
            
            # NEW: Check if domain contains brand after normalization
            # This catches cases like 'amazorn' -> 'amazom' which contains 'amazon' partially
            # We check if the normalized domain is very similar to the brand
            if len(brand) >= 4:  # Only for brands with 4+ characters
                # Check if normalized domain starts with most of the brand
                # e.g., 'amazom' starts with 'amaz' from 'amazon'
                brand_prefix = brand[:len(brand)-1]  # All but last char
                if normalized.startswith(brand_prefix) and domain_clean != brand:
                    # Also check if original doesn't start with same prefix
                    if not domain_clean.startswith(brand_prefix):
                        return True
                
                # Check for close matches using character-by-character comparison
                # This catches 'amazorn' which becomes 'amazom' (1 char different from 'amazon')
                if self._is_close_match(normalized, brand) and domain_clean != brand:
                    return True
        
        return False
    
    def _is_close_match(self, text1: str, text2: str, max_diff: int = 1) -> bool:
        """
        Check if two strings are close matches (differ by at most max_diff characters).
        
        Args:
            text1: First string
            text2: Second string
            max_diff: Maximum number of differing characters
            
        Returns:
            bool: True if strings are close matches
        """
        if abs(len(text1) - len(text2)) > max_diff:
            return False
        
        # Count differing characters
        min_len = min(len(text1), len(text2))
        diff_count = sum(1 for i in range(min_len) if text1[i] != text2[i])
        diff_count += abs(len(text1) - len(text2))  # Add length difference
        
        return diff_count <= max_diff


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

