"""
PHISHING DETECTION - EXAM CODE (20-35 lines)
Author: [Your Name]
"""

import re

class PhishingDetector:
    def __init__(self):
        self.keywords = ['verify', 'account', 'update', 'secure', 'login']
    
    def detect(self, url):
        # Extract features
        length = len(url)
        has_ip = bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url))
        keyword_count = sum(1 for k in self.keywords if k in url.lower())
        
        # Calculate phishing score
        score = 0
        if length > 75: score += 0.3
        if has_ip: score += 0.4
        score += keyword_count * 0.15
        
        # Predict
        return 'Phishing' if score >= 0.5 else 'Legitimate', score

# Test
detector = PhishingDetector()

urls = [
    'https://www.google.com',
    'http://192.168.1.1/verify-account',
    'https://secure-login-update.com/verify'
]

for url in urls:
    prediction, confidence = detector.detect(url)
    print(f"{url[:40]:40s} -> {prediction:12s} ({confidence:.0%})")
