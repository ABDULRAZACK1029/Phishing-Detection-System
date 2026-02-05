"""
PhishTank API Integration Module
Checks URLs against PhishTank's verified phishing database

PhishTank is a collaborative clearinghouse for data and information about phishing
on the Internet. This module integrates with their API to check if URLs are known
phishing sites.
"""

import requests
import time
from typing import Dict, Optional
import os


class PhishTankAPI:
    """
    PhishTank API integration for checking URLs against verified phishing database.
    
    This class provides methods to check URLs against PhishTank's database of
    verified phishing URLs. Results are cached to reduce API calls and improve
    performance.
    
    Usage:
        api = PhishTankAPI()
        result = api.check_url('http://example.com')
        if result['is_phishing']:
            print("Phishing detected!")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PhishTank API client.
        
        Args:
            api_key (str, optional): PhishTank API key for higher rate limits.
                                    Can also be set via PHISHTANK_API_KEY env var.
        """
        self.endpoint = "http://checkurl.phishtank.com/checkurl/"
        self.api_key = api_key or os.getenv('PHISHTANK_API_KEY')
        
        # Simple in-memory cache
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Statistics
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'errors': 0
        }
    
    def check_url(self, url: str, use_cache: bool = True) -> Dict:
        """
        Check if URL is in PhishTank's verified phishing database.
        
        Args:
            url (str): The URL to check
            use_cache (bool): Whether to use cached results (default: True)
        
        Returns:
            Dict containing:
                - is_phishing (bool or None): True if phishing, False if safe, None if error
                - confidence (float): 1.0 if in database, 0.0 otherwise
                - source (str): 'PhishTank'
                - phish_id (int, optional): PhishTank ID if phishing
                - verified (bool, optional): Whether manually verified
                - verified_at (str, optional): Verification timestamp
                - in_database (bool): Whether URL was found in database
                - error (str, optional): Error message if request failed
        """
        # Check cache first
        if use_cache and url in self.cache:
            cached_result, timestamp = self.cache[url]
            if time.time() - timestamp < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return cached_result
        
        # Make API request
        try:
            params = {
                'url': url,
                'format': 'json'
            }
            
            if self.api_key:
                params['app_key'] = self.api_key
            
            self.stats['api_calls'] += 1
            
            response = requests.post(
                self.endpoint,
                data=params,
                timeout=5,  # 5 second timeout
                headers={'User-Agent': 'PhishingDetectionSystem/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data['results']['in_database']:
                    # URL is in PhishTank database
                    result = {
                        'is_phishing': data['results']['valid'],
                        'confidence': 1.0,
                        'source': 'PhishTank',
                        'phish_id': data['results'].get('phish_id'),
                        'verified': data['results'].get('verified'),
                        'verified_at': data['results'].get('verified_at'),
                        'in_database': True
                    }
                else:
                    # URL not in database
                    result = {
                        'is_phishing': False,
                        'confidence': 0.0,
                        'source': 'PhishTank',
                        'in_database': False
                    }
                
                # Cache the result
                if use_cache:
                    self.cache[url] = (result, time.time())
                
                return result
            
            else:
                # Non-200 status code
                self.stats['errors'] += 1
                return {
                    'is_phishing': None,
                    'confidence': 0.0,
                    'source': 'PhishTank',
                    'error': f'HTTP {response.status_code}',
                    'in_database': False
                }
        
        except requests.Timeout:
            # Request timeout
            self.stats['errors'] += 1
            return {
                'is_phishing': None,
                'confidence': 0.0,
                'source': 'PhishTank',
                'error': 'Request timeout',
                'in_database': False
            }
        
        except requests.ConnectionError:
            # Network connection error
            self.stats['errors'] += 1
            return {
                'is_phishing': None,
                'confidence': 0.0,
                'source': 'PhishTank',
                'error': 'Connection error',
                'in_database': False
            }
        
        except Exception as e:
            # Other errors
            self.stats['errors'] += 1
            return {
                'is_phishing': None,
                'confidence': 0.0,
                'source': 'PhishTank',
                'error': str(e),
                'in_database': False
            }
    
    def clear_cache(self):
        """Clear the cache of stored results."""
        self.cache.clear()
    
    def get_stats(self) -> Dict:
        """
        Get API usage statistics.
        
        Returns:
            Dict with api_calls, cache_hits, errors counts
        """
        return self.stats.copy()


# Global API instance (singleton pattern)
_global_phishtank_api: Optional[PhishTankAPI] = None


def get_phishtank_api() -> PhishTankAPI:
    """
    Get or create the global PhishTank API instance (singleton).
    
    Returns:
        PhishTankAPI: The global API instance
    """
    global _global_phishtank_api
    if _global_phishtank_api is None:
        _global_phishtank_api = PhishTankAPI()
    return _global_phishtank_api


def check_phishtank(url: str) -> Dict:
    """
    Convenience function to check a URL using the global PhishTank API instance.
    
    Args:
        url (str): The URL to check
    
    Returns:
        Dict: PhishTank check result
    
    Example:
        >>> result = check_phishtank('http://example.com')
        >>> if result['is_phishing']:
        ...     print("Phishing detected!")
    """
    api = get_phishtank_api()
    return api.check_url(url)


if __name__ == '__main__':
    # Test the PhishTank API
    print("PhishTank API Test\n" + "=" * 50)
    
    api = PhishTankAPI()
    
    # Test URLs
    test_urls = [
        'http://www.google.com',
        'https://l.ead.me/bgaSXI',  # Known phishing from user's example
    ]
    
    for url in test_urls:
        print(f"\nChecking: {url}")
        result = api.check_url(url)
        
        if result['is_phishing'] is True:
            print(f"  ⚠️  PHISHING DETECTED")
            print(f"  Phish ID: {result.get('phish_id')}")
            print(f"  Verified: {result.get('verified')}")
        elif result['is_phishing'] is False:
            if result.get('in_database'):
                print(f"  ✓ Safe (verified by PhishTank)")
            else:
                print(f"  ? Not in PhishTank database")
        else:
            print(f"  ✗ Error: {result.get('error')}")
    
    # Show stats
    print(f"\n{'=' * 50}")
    print(f"API Statistics:")
    stats = api.get_stats()
    print(f"  API Calls: {stats['api_calls']}")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Errors: {stats['errors']}")
