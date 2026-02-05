"""
Test PhishTank API Integration and Hybrid Detection
"""

from ml.predict import predict_url_hybrid, PHISHTANK_AVAILABLE
import json

print("=" * 80)
print("PHISHTANK API + ML HYBRID DETECTION TEST")
print("=" * 80)
print()

# Check if PhishTank is available
if PHISHTANK_AVAILABLE:
    print("[OK] PhishTank API module loaded successfully")
else:
    print("[X] PhishTank API not available (requests package missing)")
    print("  Install with: pip install requests")
    exit(1)

print()

# Test URLs
test_cases = [
    {
        'url': 'https://l.ead.me/bgaSXI',
        'description': 'Known PhishTank phishing URL',
        'expected': 'Phishing'
    },
    {
        'url': 'http://www.faceb00k.com',
        'description': 'Homoglyph attack (not in PhishTank)',
        'expected': 'Phishing'
    },
    {
        'url': 'https://bit.ly/test123',
        'description': 'URL shortener (not in PhishTank)',
        'expected': 'Phishing'
    },
    {
        'url': 'https://www.google.com',
        'description': 'Legitimate website',
        'expected': 'Legitimate'
    }
]

print("Testing Hybrid Detection (PhishTank + ML):")
print("-" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['description']}")
    print(f"URL: {test['url']}")
    
    # Run hybrid prediction
    result = predict_url_hybrid(test['url'])
    
    # Display results
    print(f"  Prediction:       {result['prediction']}")
    print(f"  Confidence:       {result['confidence']:.0%}")
    print(f"  Threat Level:     {result['threat_level'].upper()}")
    print(f"  Detection Method: {result.get('detection_method', 'N/A')}")
    print(f"  Source:           {result.get('source', 'N/A')}")
    
    # PhishTank specific info
    if result.get('phish_id'):
        print(f"  PhishTank ID:     {result['phish_id']}")
        print(f"  Verified:         {result.get('verified', 'N/A')}")
    
    # Check if matches expected
    if result['prediction'] == test['expected']:
        print(f"  Result:           [PASS]")
    else:
        print(f"  Result:           [FAIL] (expected {test['expected']})")

print()
print("=" * 80)

# Test with PhishTank disabled (ML only)
print("\nTesting ML-Only Mode (PhishTank disabled):")
print("-" * 80)

test_url = 'http://www.faceb00k.com'
result = predict_url_hybrid(test_url, use_phishtank=False)

print(f"URL: {test_url}")
print(f"  Prediction:       {result['prediction']}")
print(f"  Confidence:       {result['confidence']:.0%}")
print(f"  Source:           {result.get('source', 'N/A')}")
print(f"  Detection Method: {result.get('detection_method', 'N/A')}")

print()
print("=" * 80)
print("Test Complete!")
print("=" * 80)
