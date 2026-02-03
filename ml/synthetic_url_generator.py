"""
Synthetic Phishing URL Generator
==================================

This module generates realistic phishing-style URLs for ethical testing and 
academic research in cybersecurity education (BCA Final Year Project).

IMPORTANT ETHICAL NOTICE:
-------------------------
All synthetic URLs use SAFE TLDs (.test, .example, .invalid) as defined in 
RFC 2606 and RFC 6761. These TLDs are reserved and will NEVER be registered 
as real domains, ensuring:
  - No real websites are affected
  - No legal issues with domain squatting
  - Compliance with ethical research practices
  - Academic integrity in cybersecurity education

This code is designed for educational purposes to understand attacker 
techniques and test phishing detection models in a safe, controlled environment.

Author: BCA Final Year Project - Phishing Detection System
License: Educational Use Only
"""

import random
import string
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RealismLevel(Enum):
    """Realism level for synthetic phishing URLs."""
    LOW = "low"          # Obvious phishing, multiple red flags
    MEDIUM = "medium"    # Moderate sophistication
    HIGH = "high"        # Advanced, subtle phishing


class BrandCategory(Enum):
    """Categories of brands to impersonate."""
    BANKING = "banking"
    ECOMMERCE = "ecommerce"
    SOCIAL = "social"
    CLOUD = "cloud"
    TECH = "tech"


@dataclass
class BrandInfo:
    """Information about a brand for impersonation."""
    name: str
    category: BrandCategory
    common_subdomains: List[str]
    common_paths: List[str]


class BrandDatabase:
    """
    Database of legitimate brands for synthetic phishing URL generation.
    
    This class stores information about popular brands across different
    categories to enable realistic brand impersonation in synthetic URLs.
    """
    
    def __init__(self):
        """Initialize the brand database."""
        self.brands = {
            BrandCategory.BANKING: [
                BrandInfo("paypal", BrandCategory.BANKING, 
                         ["secure", "login", "account", "verify"],
                         ["/signin", "/verify", "/account", "/security"]),
                BrandInfo("chase", BrandCategory.BANKING,
                         ["secure", "online", "banking", "login"],
                         ["/login", "/account", "/verify", "/secure"]),
                BrandInfo("bankofamerica", BrandCategory.BANKING,
                         ["secure", "online", "myaccounts"],
                         ["/login", "/signin", "/verify"]),
                BrandInfo("wellsfargo", BrandCategory.BANKING,
                         ["secure", "online", "accounts"],
                         ["/login", "/verify", "/account"]),
            ],
            BrandCategory.ECOMMERCE: [
                BrandInfo("amazon", BrandCategory.ECOMMERCE,
                         ["secure", "signin", "account", "www"],
                         ["/signin", "/account", "/verify", "/ap/signin"]),
                BrandInfo("ebay", BrandCategory.ECOMMERCE,
                         ["signin", "account", "secure"],
                         ["/signin", "/account", "/verify"]),
                BrandInfo("alibaba", BrandCategory.ECOMMERCE,
                         ["login", "secure", "account"],
                         ["/login", "/account", "/verify"]),
                BrandInfo("walmart", BrandCategory.ECOMMERCE,
                         ["secure", "account", "login"],
                         ["/login", "/account", "/signin"]),
            ],
            BrandCategory.SOCIAL: [
                BrandInfo("facebook", BrandCategory.SOCIAL,
                         ["secure", "login", "m", "www"],
                         ["/login", "/recover", "/verify"]),
                BrandInfo("twitter", BrandCategory.SOCIAL,
                         ["secure", "login", "mobile"],
                         ["/login", "/sessions", "/account"]),
                BrandInfo("instagram", BrandCategory.SOCIAL,
                         ["secure", "login", "account"],
                         ["/accounts/login", "/verify", "/recover"]),
                BrandInfo("linkedin", BrandCategory.SOCIAL,
                         ["secure", "login", "www"],
                         ["/login", "/checkpoint", "/verify"]),
            ],
            BrandCategory.CLOUD: [
                BrandInfo("google", BrandCategory.CLOUD,
                         ["accounts", "mail", "drive", "secure"],
                         ["/signin", "/ServiceLogin", "/verify"]),
                BrandInfo("microsoft", BrandCategory.CLOUD,
                         ["login", "account", "outlook", "office"],
                         ["/login", "/common/oauth2", "/verify"]),
                BrandInfo("dropbox", BrandCategory.CLOUD,
                         ["secure", "login", "account"],
                         ["/login", "/account", "/verify"]),
                BrandInfo("icloud", BrandCategory.CLOUD,
                         ["secure", "login", "account"],
                         ["/signin", "/account", "/verify"]),
            ],
            BrandCategory.TECH: [
                BrandInfo("apple", BrandCategory.TECH,
                         ["secure", "appleid", "account"],
                         ["/account", "/signin", "/verify"]),
                BrandInfo("adobe", BrandCategory.TECH,
                         ["secure", "account", "login"],
                         ["/signin", "/account", "/verify"]),
                BrandInfo("oracle", BrandCategory.TECH,
                         ["login", "secure", "account"],
                         ["/signin", "/sso", "/verify"]),
                BrandInfo("samsung", BrandCategory.TECH,
                         ["account", "secure", "login"],
                         ["/account/login", "/verify", "/signin"]),
            ]
        }
    
    def get_brands(self, category: Optional[BrandCategory] = None) -> List[BrandInfo]:
        """Get brands by category or all brands."""
        if category:
            return self.brands.get(category, [])
        return [brand for brands in self.brands.values() for brand in brands]
    
    def get_random_brand(self, category: Optional[BrandCategory] = None) -> BrandInfo:
        """Get a random brand from specified category or all categories."""
        brands = self.get_brands(category)
        return random.choice(brands)


class AttackerPatterns:
    """
    Implementation of common attacker techniques for URL manipulation.
    
    This class implements various typosquatting and obfuscation techniques
    used by real-world phishing attackers.
    """
    
    # Character substitution mappings (homoglyphs)
    CHAR_SUBSTITUTIONS = {
        'a': ['@', '4'],
        'e': ['3'],
        'i': ['1', '!', 'l'],
        'o': ['0'],
        's': ['5', '$'],
        'l': ['1', 'i'],
        'g': ['9'],
        't': ['7'],
    }
    
    # Urgency and security keywords
    URGENCY_KEYWORDS = [
        "urgent", "immediate", "alert", "warning", "suspended", "locked",
        "verify", "confirm", "update", "secure", "action-required",
        "limited-time", "expires", "important", "critical", "attention"
    ]
    
    SECURITY_KEYWORDS = [
        "secure", "security", "verified", "protected", "safe", "ssl",
        "https", "encrypted", "auth", "authentication", "validation"
    ]
    
    @staticmethod
    def typosquat_substitution(domain: str, intensity: int = 1) -> str:
        """
        Apply character substitution typosquatting.
        
        Args:
            domain: Original domain name
            intensity: Number of substitutions to apply (1-3)
        
        Returns:
            Typosquatted domain
        """
        domain_list = list(domain)
        substitutions_made = 0
        
        for i, char in enumerate(domain_list):
            if char.lower() in AttackerPatterns.CHAR_SUBSTITUTIONS:
                if random.random() < 0.3 and substitutions_made < intensity:
                    replacements = AttackerPatterns.CHAR_SUBSTITUTIONS[char.lower()]
                    domain_list[i] = random.choice(replacements)
                    substitutions_made += 1
        
        return ''.join(domain_list)
    
    @staticmethod
    def typosquat_omission(domain: str) -> str:
        """Remove a random character (character omission)."""
        if len(domain) <= 3:
            return domain
        pos = random.randint(1, len(domain) - 2)
        return domain[:pos] + domain[pos + 1:]
    
    @staticmethod
    def typosquat_insertion(domain: str) -> str:
        """Insert a random character (character insertion)."""
        pos = random.randint(1, len(domain) - 1)
        char = random.choice(string.ascii_lowercase)
        return domain[:pos] + char + domain[pos:]
    
    @staticmethod
    def typosquat_transposition(domain: str) -> str:
        """Swap adjacent characters (transposition)."""
        if len(domain) <= 2:
            return domain
        pos = random.randint(0, len(domain) - 2)
        domain_list = list(domain)
        domain_list[pos], domain_list[pos + 1] = domain_list[pos + 1], domain_list[pos]
        return ''.join(domain_list)
    
    @staticmethod
    def typosquat_repetition(domain: str) -> str:
        """Repeat a random character (character repetition)."""
        pos = random.randint(0, len(domain) - 1)
        return domain[:pos] + domain[pos] + domain[pos:]
    
    @staticmethod
    def apply_random_typosquat(domain: str, realism: RealismLevel) -> str:
        """
        Apply random typosquatting technique based on realism level.
        
        Args:
            domain: Original domain
            realism: Realism level (affects probability and intensity)
        
        Returns:
            Typosquatted domain
        """
        techniques = [
            AttackerPatterns.typosquat_substitution,
            AttackerPatterns.typosquat_omission,
            AttackerPatterns.typosquat_insertion,
            AttackerPatterns.typosquat_transposition,
            AttackerPatterns.typosquat_repetition,
        ]
        
        # High realism = less obvious typos
        if realism == RealismLevel.HIGH:
            # Only subtle substitutions
            return AttackerPatterns.typosquat_substitution(domain, intensity=1)
        elif realism == RealismLevel.MEDIUM:
            # 1-2 techniques
            technique = random.choice(techniques)
            return technique(domain)
        else:  # LOW
            # Multiple techniques, obvious
            result = domain
            for _ in range(random.randint(1, 2)):
                technique = random.choice(techniques)
                result = technique(result)
            return result


class SyntheticURLGenerator:
    """
    Main generator for synthetic phishing and legitimate URLs.
    
    This class orchestrates the generation of realistic phishing-style URLs
    using various attacker patterns while ensuring all URLs use safe TLDs.
    """
    
    # Safe TLDs (RFC 2606 and RFC 6761)
    SAFE_TLDS = [".test", ".example", ".invalid"]
    
    # Additional suspicious TLDs for variety (but still using safe base)
    SUSPICIOUS_TLDS = [".test", ".example", ".invalid"]
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the synthetic URL generator.
        
        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            random.seed(seed)
        
        self.brand_db = BrandDatabase()
        self.patterns = AttackerPatterns()
    
    def generate_phishing_url(
        self,
        category: Optional[BrandCategory] = None,
        realism: RealismLevel = RealismLevel.MEDIUM
    ) -> str:
        """
        Generate a single synthetic phishing URL.
        
        Args:
            category: Brand category to impersonate (random if None)
            realism: Realism level for the phishing URL
        
        Returns:
            Synthetic phishing URL using safe TLD
        """
        # Select a brand to impersonate
        brand = self.brand_db.get_random_brand(category)
        
        # Choose attack pattern
        pattern_type = random.choice([
            "typosquat",
            "subdomain_brand",
            "hyphenated",
            "subdomain_keywords",
            "mixed"
        ])
        
        if pattern_type == "typosquat":
            # Typosquatted brand name
            domain = self.patterns.apply_random_typosquat(brand.name, realism)
            subdomain = None
        
        elif pattern_type == "subdomain_brand":
            # Brand in subdomain, suspicious main domain
            subdomain = brand.name
            domain = random.choice(["secure", "login", "verify", "account", "update"])
        
        elif pattern_type == "hyphenated":
            # Hyphenated keywords with brand
            keywords = random.sample(self.patterns.SECURITY_KEYWORDS + 
                                   self.patterns.URGENCY_KEYWORDS, 
                                   k=random.randint(2, 3))
            domain = "-".join(keywords + [brand.name])
            subdomain = None
        
        elif pattern_type == "subdomain_keywords":
            # Multiple suspicious subdomains
            domain = brand.name
            subdomain_parts = random.sample(
                brand.common_subdomains + self.patterns.URGENCY_KEYWORDS,
                k=random.randint(2, 4)
            )
            subdomain = ".".join(subdomain_parts)
        
        else:  # mixed
            # Combination of techniques
            domain = self.patterns.apply_random_typosquat(brand.name, realism)
            subdomain = random.choice(brand.common_subdomains + ["secure", "verify"])
        
        # Select safe TLD
        tld = random.choice(self.SAFE_TLDS)
        
        # Construct domain
        if subdomain:
            full_domain = f"{subdomain}.{domain}{tld}"
        else:
            full_domain = f"{domain}{tld}"
        
        # Add path (common in phishing)
        path = random.choice(brand.common_paths + ["", ""])
        
        # Add query parameters (sometimes)
        query = ""
        if random.random() < 0.4:
            params = []
            if random.random() < 0.5:
                params.append(f"user={random.randint(1000, 9999)}")
            if random.random() < 0.5:
                params.append(f"token={''.join(random.choices(string.ascii_lowercase + string.digits, k=16))}")
            if random.random() < 0.3:
                params.append(f"redirect={''.join(random.choices(string.ascii_lowercase, k=8))}")
            
            if params:
                query = "?" + "&".join(params)
        
        # Add port (sometimes for low realism)
        port = ""
        if realism == RealismLevel.LOW and random.random() < 0.3:
            port = f":{random.choice([8080, 8443, 8888, 3000])}"
        
        # Construct full URL
        url = f"https://{full_domain}{port}{path}{query}"
        
        return url
    
    def generate_legitimate_url(
        self,
        category: Optional[BrandCategory] = None
    ) -> str:
        """
        Generate a synthetic legitimate-looking URL.
        
        Args:
            category: Brand category (random if None)
        
        Returns:
            Synthetic legitimate URL using safe TLD
        """
        brand = self.brand_db.get_random_brand(category)
        
        # Legitimate URLs are simpler
        subdomain = random.choice(["www", ""] + brand.common_subdomains[:2])
        domain = brand.name
        tld = random.choice(self.SAFE_TLDS)
        
        if subdomain:
            full_domain = f"{subdomain}.{domain}{tld}"
        else:
            full_domain = f"{domain}{tld}"
        
        # Simple path or no path
        path = random.choice(["", "/", "/home", "/about"] + brand.common_paths[:1])
        
        url = f"https://{full_domain}{path}"
        
        return url
    
    def generate_phishing_urls(
        self,
        count: int,
        categories: Optional[List[BrandCategory]] = None,
        realism: RealismLevel = RealismLevel.MEDIUM
    ) -> List[str]:
        """
        Generate multiple phishing URLs.
        
        Args:
            count: Number of URLs to generate
            categories: List of categories to use (all if None)
            realism: Realism level
        
        Returns:
            List of synthetic phishing URLs
        """
        urls = []
        for _ in range(count):
            category = random.choice(categories) if categories else None
            url = self.generate_phishing_url(category, realism)
            urls.append(url)
        return urls
    
    def generate_legitimate_urls(
        self,
        count: int,
        categories: Optional[List[BrandCategory]] = None
    ) -> List[str]:
        """
        Generate multiple legitimate URLs.
        
        Args:
            count: Number of URLs to generate
            categories: List of categories to use (all if None)
        
        Returns:
            List of synthetic legitimate URLs
        """
        urls = []
        for _ in range(count):
            category = random.choice(categories) if categories else None
            url = self.generate_legitimate_url(category)
            urls.append(url)
        return urls
    
    def generate_balanced_dataset(
        self,
        total_count: int,
        categories: Optional[List[BrandCategory]] = None,
        realism: RealismLevel = RealismLevel.MEDIUM
    ) -> Tuple[List[str], List[int]]:
        """
        Generate a balanced dataset of phishing and legitimate URLs.
        
        Args:
            total_count: Total number of URLs to generate
            categories: List of categories to use (all if None)
            realism: Realism level for phishing URLs
        
        Returns:
            Tuple of (urls, labels) where labels are 0=legitimate, 1=phishing
        """
        phishing_count = total_count // 2
        legitimate_count = total_count - phishing_count
        
        phishing_urls = self.generate_phishing_urls(phishing_count, categories, realism)
        legitimate_urls = self.generate_legitimate_urls(legitimate_count, categories)
        
        # Combine and create labels
        urls = phishing_urls + legitimate_urls
        labels = [1] * len(phishing_urls) + [0] * len(legitimate_urls)
        
        # Shuffle
        combined = list(zip(urls, labels))
        random.shuffle(combined)
        urls, labels = zip(*combined)
        
        return list(urls), list(labels)


# Convenience functions
def generate_synthetic_phishing_urls(
    count: int = 100,
    realism: str = "medium",
    categories: Optional[List[str]] = None
) -> List[str]:
    """
    Convenience function to generate phishing URLs.
    
    Args:
        count: Number of URLs to generate
        realism: "low", "medium", or "high"
        categories: List of category names (e.g., ["banking", "ecommerce"])
    
    Returns:
        List of synthetic phishing URLs
    """
    generator = SyntheticURLGenerator()
    realism_level = RealismLevel(realism)
    
    category_objs = None
    if categories:
        category_objs = [BrandCategory(cat) for cat in categories]
    
    return generator.generate_phishing_urls(count, category_objs, realism_level)


def generate_synthetic_dataset(
    count: int = 1000,
    realism: str = "medium"
) -> Tuple[List[str], List[int]]:
    """
    Convenience function to generate balanced dataset.
    
    Args:
        count: Total number of URLs
        realism: "low", "medium", or "high"
    
    Returns:
        Tuple of (urls, labels)
    """
    generator = SyntheticURLGenerator()
    realism_level = RealismLevel(realism)
    return generator.generate_balanced_dataset(count, realism=realism_level)


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("SYNTHETIC PHISHING URL GENERATOR - DEMO")
    print("=" * 70)
    print("\nETHICAL NOTICE: All URLs use safe TLDs (.test, .example, .invalid)")
    print("These domains are reserved and will never be registered.\n")
    
    generator = SyntheticURLGenerator(seed=42)
    
    # Generate sample phishing URLs
    print("\n" + "-" * 70)
    print("SAMPLE PHISHING URLs (Medium Realism)")
    print("-" * 70)
    phishing_urls = generator.generate_phishing_urls(5, realism=RealismLevel.MEDIUM)
    for i, url in enumerate(phishing_urls, 1):
        print(f"{i}. {url}")
    
    # Generate sample legitimate URLs
    print("\n" + "-" * 70)
    print("SAMPLE LEGITIMATE URLs")
    print("-" * 70)
    legitimate_urls = generator.generate_legitimate_urls(5)
    for i, url in enumerate(legitimate_urls, 1):
        print(f"{i}. {url}")
    
    # Generate balanced dataset
    print("\n" + "-" * 70)
    print("BALANCED DATASET GENERATION")
    print("-" * 70)
    urls, labels = generator.generate_balanced_dataset(20)
    print(f"Generated {len(urls)} URLs")
    print(f"Phishing: {sum(labels)} | Legitimate: {len(labels) - sum(labels)}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
