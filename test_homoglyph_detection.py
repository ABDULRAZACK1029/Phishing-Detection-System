"""
Test script for homoglyph detection
Tests ASCII-based homoglyph attacks like amazorn.com (rn->m) and faceb00k.com (0->o)
"""

from ml.predict import predict_url

def test_homoglyph_detection():
    """Test homoglyph detection with various phishing URLs."""
    
    print("=" * 80)
    print("HOMOGLYPH DETECTION TEST")
    print("=" * 80)
    print()
    
    # Test cases: (URL, expected_is_safe, description)
    test_cases = [
        # Homoglyph attacks - should be detected as phishing
        ('http://www.amazorn.com/orders', False, 'Amazon with rn->m substitution'),
        ('http://www.amaz0n.com/login', False, 'Amazon with 0->o substitution'),
        ('http://www.faceb00k-login.com', False, 'Facebook with 0->o substitution'),
        ('http://www.faceb00k.com', False, 'Facebook with 0->o substitution'),
        ('http://www.paypa1.com', False, 'PayPal with 1->l substitution'),
        ('http://www.paypa1-secure.com', False, 'PayPal with 1->l and keywords'),
        ('http://www.g00gle.com', False, 'Google with 0->o substitution'),
        ('http://www.micr0soft.com', False, 'Microsoft with 0->o substitution'),
        ('http://www.app1e.com', False, 'Apple with 1->l substitution'),
        ('http://www.netf1ix.com', False, 'Netflix with 1->l substitution'),
        
        # Legitimate URLs - should be safe
        ('http://www.amazon.com', True, 'Legitimate Amazon'),
        ('http://www.facebook.com', True, 'Legitimate Facebook'),
        ('http://www.paypal.com', True, 'Legitimate PayPal'),
        ('http://www.google.com', True, 'Legitimate Google'),
        ('http://www.microsoft.com', True, 'Legitimate Microsoft'),
        ('http://www.apple.com', True, 'Legitimate Apple'),
        
        # Non-brand URLs with numbers - should be safe (no brand impersonation)
        ('http://www.example123.com', True, 'Generic domain with numbers'),
        ('http://www.test0site.com', True, 'Generic domain with zero'),
    ]
    
    passed = 0
    failed = 0
    
    for url, expected_safe, description in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Test: {description}")
        print(f"URL:  {url}")
        print(f"Expected: {'SAFE' if expected_safe else 'PHISHING'}")
        
        result = predict_url(url)
        
        actual_safe = result['is_safe']
        print(f"Actual:   {'SAFE' if actual_safe else 'PHISHING'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Threat Level: {result['threat_level'].upper()}")
        
        if 'detection_method' in result:
            print(f"Detection Method: {result['detection_method']}")
        
        # Check if test passed
        if actual_safe == expected_safe:
            print("[PASS]")
            passed += 1
        else:
            print("[FAIL]")
            failed += 1
            
            # Show feature details for failed tests
            if result['features'].get('has_ascii_homoglyph'):
                print(f"   has_ascii_homoglyph: {result['features']['has_ascii_homoglyph']}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed} [PASS]")
    print(f"Failed: {failed} [FAIL]")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print()
    
    if failed == 0:
        print("SUCCESS: All tests passed! Homoglyph detection is working correctly.")
    else:
        print("WARNING: Some tests failed. Review the output above for details.")
    
    return failed == 0


if __name__ == '__main__':
    success = test_homoglyph_detection()
    exit(0 if success else 1)
