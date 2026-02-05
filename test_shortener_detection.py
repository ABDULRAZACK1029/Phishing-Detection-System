"""Test URL shortener detection"""
from ml.predict import predict_url

# Test the PhishTank URL
url = 'https://l.ead.me/bgaSXI'
result = predict_url(url)

print("=" * 70)
print("URL SHORTENER DETECTION TEST")
print("=" * 70)
print()
print(f"URL: {url}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Threat Level: {result['threat_level'].upper()}")
print(f"Detection Method: {result.get('detection_method', 'N/A')}")
print(f"Is Shortened: {result['features']['is_shortened']}")
print()

# Test other shorteners
test_urls = [
    'https://bit.ly/3xK9a',
    'https://t.co/abc123',
    'https://short.io/xYz',
    'https://www.google.com',
]

print("Additional Tests:")
print("-" * 70)
for test_url in test_urls:
    result = predict_url(test_url)
    print(f"{test_url:40s} -> {result['prediction']:12s} ({result['confidence']:.0%})")
