from ml.predict import predict_url

# Test multiple phishing URL patterns
test_urls = [
    'https://whatsapp-account-suspended.verify.test',
    'https://paypal-secure-login-verify.example',
    'https://facebook-security-alert.invalid',
    'https://amazon-account-locked.verify.test',
    'https://google-verify-account.suspended.example',
    'https://microsoft-urgent-update.test',
    'https://instagram-confirm-identity.example',
    'https://bank-security-update-required.test',
]

print("=" * 70)
print("COMPREHENSIVE PHISHING URL DETECTION TEST")
print("=" * 70)

correct_detections = 0
total_tests = len(test_urls)

for i, url in enumerate(test_urls, 1):
    print(f"\n[Test {i}/{total_tests}] {url}")
    
    result = predict_url(url)
    
    if result['is_safe']:
        print(f"  [FAIL] Detected as {result['prediction']} ({result['confidence']:.0%} confidence)")
    else:
        print(f"  [PASS] Detected as {result['prediction']} ({result['confidence']:.0%} confidence, {result['threat_level'].upper()} threat)")
        correct_detections += 1

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Correct Detections: {correct_detections}/{total_tests} ({correct_detections/total_tests*100:.1f}%)")

if correct_detections == total_tests:
    print("[SUCCESS] ALL TESTS PASSED - Phishing detection is working correctly!")
else:
    print(f"[WARNING] {total_tests - correct_detections} test(s) failed")

print("=" * 70)
