"""
Multi-Dataset Loader for Phishing Detection
Downloads and combines multiple phishing datasets for robust ML training
"""

import os
import pandas as pd
import numpy as np
import requests
from typing import Tuple, List, Dict
import warnings
from pathlib import Path

# Import feature extractor
try:
    from ml.feature_extraction import FeatureExtractor
except ImportError:
    from feature_extraction import FeatureExtractor


class MultiDatasetLoader:
    """
    Loads and combines phishing datasets from multiple sources:
    - PhiUSIIL Dataset (UCI) - 235K URLs
    - PhishTank Live Data - Active phishing URLs
    - Kaggle Crawling2024 - 56K URLs
    
    Handles different CSV formats and extracts features uniformly.
    """
    
    def __init__(self, cache_dir: str = 'ml/datasets'):
        """
        Initialize the multi-dataset loader.
        
        Args:
            cache_dir (str): Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        
    def download_phiusiil_dataset(self) -> pd.DataFrame:
        """
        Download PhiUSIIL dataset from UCI ML Repository.
        
        This is the primary dataset with 235K URLs (134,850 legitimate + 100,945 phishing).
        The dataset can be loaded using ucimlrepo package or downloaded as CSV.
        
        Returns:
            pd.DataFrame: Dataset with 'url' and 'label' columns
        """
        cache_file = self.cache_dir / 'phiusiil_dataset.csv'
        
        if cache_file.exists():
            print(f"[OK] Loading cached PhiUSIIL dataset from {cache_file}")
            return pd.read_csv(cache_file)
        
        print("[*] Downloading PhiUSIIL dataset from UCI...")
        
        try:
            # Try using ucimlrepo package first
            from ucimlrepo import fetch_ucirepo
            
            # Fetch dataset (ID: 967 for PhiUSIIL)
            phishing_dataset = fetch_ucirepo(id=967)
            
            # Extract features and targets
            X = phishing_dataset.data.features
            y = phishing_dataset.data.targets
            
            # The dataset has 'URL' column and target label
            df = pd.DataFrame(X)
            df['label'] = y
            
            # Rename URL column if necessary
            url_col = [col for col in df.columns if 'url' in col.lower()][0]
            df = df.rename(columns={url_col: 'url'})
            
            # Keep only url and label
            df = df[['url', 'label']]
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            print(f"[OK] Downloaded {len(df)} URLs from PhiUSIIL dataset")
            
            return df
            
        except ImportError:
            print("[!] ucimlrepo package not found. Please install: pip install ucimlrepo")
            print("Creating empty placeholder - you'll need to manually download the dataset")
            
            # Create empty placeholder
            df = pd.DataFrame(columns=['url', 'label'])
            return df
        except Exception as e:
            print(f"[!] Error downloading PhiUSIIL dataset: {e}")
            print("Creating empty placeholder")
            df = pd.DataFrame(columns=['url', 'label'])
            return df
    
    
    def load_mendeley_dataset(self) -> pd.DataFrame:
        """
        Load Mendeley 'Phishing vs Legitimate URLs' dataset (vfszbj9b36).
        Expected filename: ml/datasets/mendeley_dataset.csv
        """
        cache_file = self.cache_dir / 'mendeley_dataset.csv'
        
        if not cache_file.exists():
            print("[-] Mendeley dataset not found in ml/datasets/mendeley_dataset.csv")
            return pd.DataFrame(columns=['url', 'label'])
            
        print(f"[*] Loading Mendeley dataset from {cache_file}...")
        try:
            # Try reading with different encodings just in case
            try:
                df = pd.read_csv(cache_file)
            except UnicodeDecodeError:
                df = pd.read_csv(cache_file, encoding='latin1')
                
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            
            # Identify columns
            url_col = next((c for c in df.columns if 'url' in c), None)
            label_col = next((c for c in df.columns if 'status' in c or 'label' in c or 'type' in c), None)
            
            if not url_col or not label_col:
                print(f"[!] Could not identify URL/Label columns in {df.columns}")
                return pd.DataFrame(columns=['url', 'label'])
            
            # Normalize DataFrame
            df = df.rename(columns={url_col: 'url', label_col: 'label'})
            df = df[['url', 'label']]
            
            # Map labels to 0/1 if they are strings (legitimate/phishing)
            # Mendeley dataset often has 'legitimate' and 'phishing' strings
            if df['label'].dtype == 'O': 
                df['label'] = df['label'].map({
                    'legitimate': 0, 'Legitimate': 0, '0': 0, 0: 0,
                    'phishing': 1, 'Phishing': 1, '1': 1, 1: 1
                })
            
            # Drop unmapped rows
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)
            
            print(f"[OK] Loaded {len(df)} URLs from Mendeley dataset")
            return df
            
        except Exception as e:
            print(f"[!] Error loading Mendeley dataset: {e}")
            return pd.DataFrame(columns=['url', 'label'])

    def download_phishtank_dataset(self, limit: int = 10000) -> pd.DataFrame:
        """
        Download live phishing data from PhishTank.
        
        PhishTank provides hourly updated phishing URLs.
        Due to large size, we limit to most recent entries.
        
        Args:
            limit (int): Maximum number of URLs to download
            
        Returns:
            pd.DataFrame: Dataset with 'url' and 'label' columns
        """
        cache_file = self.cache_dir / 'phishtank_dataset.csv'
        
        if cache_file.exists():
            print(f"[OK] Loading cached PhishTank dataset from {cache_file}")
            return pd.read_csv(cache_file)
        
        print("[*] Downloading PhishTank live data...")
        
        try:
            # Download from PhishTank public API
            url = "http://data.phishtank.com/data/online-valid.csv"
            
            # Download with timeout
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save raw CSV
            temp_file = self.cache_dir / 'phishtank_raw.csv'
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Parse CSV
            df = pd.read_csv(temp_file)
            
            # PhishTank CSV has 'url' column and 'verified' column
            # All URLs in this dataset are phishing
            if 'url' in df.columns:
                df = df[['url']].copy()
                df['label'] = 1  # All phishing
                
                # Limit to avoid huge dataset
                if len(df) > limit:
                    df = df.sample(n=limit, random_state=42)
                
                # Save to cache
                df.to_csv(cache_file, index=False)
                print(f"[OK] Downloaded {len(df)} phishing URLs from PhishTank")
                
                # Clean up temp file
                temp_file.unlink()
                
                return df
            else:
                print("[!] Unexpected PhishTank format")
                return pd.DataFrame(columns=['url', 'label'])
                
        except Exception as e:
            print(f"[!] Error downloading PhishTank data: {e}")
            print("Skipping PhishTank dataset - continuing with other sources")
            return pd.DataFrame(columns=['url', 'label'])

    def download_openphish_dataset(self, limit: int = 10000) -> pd.DataFrame:
        """
        Download live phishing URLs from OpenPhish free feed.

        OpenPhish provides a continuously updated free feed of phishing URLs
        at https://openphish.com/feed.txt — one URL per line, all confirmed phishing.

        Args:
            limit (int): Max number of URLs to use (default 10,000)

        Returns:
            pd.DataFrame: Dataset with 'url' and 'label' columns (all label=1)
        """
        cache_file = self.cache_dir / 'openphish_dataset.csv'

        if cache_file.exists():
            print(f"[OK] Loading cached OpenPhish dataset from {cache_file}")
            return pd.read_csv(cache_file)

        print("[*] Downloading OpenPhish live phishing feed...")

        try:
            response = requests.get(
                'https://openphish.com/feed.txt',
                timeout=15,
                headers={'User-Agent': 'PhishGuard/1.0 Academic Research'}
            )
            response.raise_for_status()

            # Each line is one phishing URL
            urls = [line.strip() for line in response.text.splitlines()
                    if line.strip() and line.startswith('http')]

            if not urls:
                print("[!] OpenPhish feed returned no URLs")
                return pd.DataFrame(columns=['url', 'label'])

            # Limit if needed
            if len(urls) > limit:
                import random
                random.seed(42)
                urls = random.sample(urls, limit)

            df = pd.DataFrame({'url': urls, 'label': 1})  # All phishing

            # Cache for future runs
            df.to_csv(cache_file, index=False)
            print(f"[OK] Downloaded {len(df)} phishing URLs from OpenPhish")
            return df

        except requests.Timeout:
            print("[!] OpenPhish request timed out — skipping")
            return pd.DataFrame(columns=['url', 'label'])
        except Exception as e:
            print(f"[!] Error downloading OpenPhish data: {e}")
            return pd.DataFrame(columns=['url', 'label'])
    
    def create_legitimate_urls(self, count: int = 10000) -> pd.DataFrame:
        """
        Create legitimate URL dataset from known safe sources.
        
        PhishTank only provides phishing URLs, so we need legitimate URLs
        to balance the dataset.
        
        Args:
            count (int): Number of legitimate URLs to generate
            
        Returns:
            pd.DataFrame: Dataset with 'url' and 'label' columns
        """
        cache_file = self.cache_dir / 'legitimate_urls.csv'
        
        if cache_file.exists():
            print(f"[OK] Loading cached legitimate URLs from {cache_file}")
            return pd.read_csv(cache_file)
        
        print(f"[*] Creating {count} legitimate URLs...")
        
        # Popular legitimate domains
        legitimate_domains = [
            # Tech companies
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
            'instagram.com', 'linkedin.com', 'microsoft.com', 'apple.com',
            'amazon.com', 'ebay.com', 'netflix.com', 'github.com',
            
            # Educational
            'wikipedia.org', 'stackoverflow.com', 'reddit.com', 'medium.com',
            
            # News
            'nytimes.com', 'bbc.com', 'cnn.com', 'theguardian.com',
            
            # Financial
            'paypal.com', 'chase.com', 'wellsfargo.com', 'bankofamerica.com',
            
            # E-commerce
            'shopify.com', 'walmart.com', 'target.com', 'bestbuy.com',
        ]
        
        # Common paths
        paths = [
            '', '/', '/about', '/contact', '/help', '/support', '/faq',
            '/products', '/services', '/pricing', '/login', '/signup',
            '/blog', '/news', '/search', '/profile', '/settings',
            '/dashboard', '/home', '/index.html', '/api/v1/users'
        ]
        
        urls = []
        for _ in range(count):
            domain = np.random.choice(legitimate_domains)
            path = np.random.choice(paths)
            protocol = np.random.choice(['https', 'http'], p=[0.9, 0.1])
            
            url = f"{protocol}://{domain}{path}"
            
            # Add query parameters sometimes
            if np.random.random() < 0.2:
                url += f"?page={np.random.randint(1, 100)}"
            
            urls.append(url)
        
        df = pd.DataFrame({
            'url': urls,
            'label': 0  # Legitimate
        })
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        print(f"[OK] Created {len(df)} legitimate URLs")
        
        return df
    
    def extract_features_from_urls(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ML features from URLs using FeatureExtractor.
        
        Args:
            df (pd.DataFrame): DataFrame with 'url' and 'label' columns
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix X and labels y
        """
        print(f"[*] Extracting features from {len(df)} URLs...")
        
        X = []
        y = []
        errors = 0
        
        for idx, row in df.iterrows():
            try:
                url = row['url']
                label = row['label']
                
                # Extract features
                features = self.feature_extractor.extract_features(url)
                
                # Convert to array in correct order
                feature_vector = [
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
                    features['entropy'],
                    features['digit_ratio'],
                    features['longest_token_len'],
                    features['tld_in_path'],
                    features['is_shortened'],
                    features['is_suspicious_tld'],
                    features['has_client_server'],
                    features['domain_token_count'],
                    features['path_token_count'],
                    features['is_punycode']
                ]
                
                X.append(feature_vector)
                y.append(label)
                
            except Exception as e:
                errors += 1
                if errors < 10:  # Show first 10 errors only
                    print(f"[!] Error processing URL {idx}: {e}")
                continue
            
            # Progress indicator
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1}/{len(df)} URLs...")
        
        if errors > 0:
            print(f"[!] Skipped {errors} URLs due to errors")
        
        print(f"[OK] Extracted features from {len(X)} URLs")
        
        return np.array(X), np.array(y)
    
    def load_and_combine_datasets(self,
                                  use_phiusiil: bool = True,
                                  use_phishtank: bool = True,
                                  use_mendeley: bool = True,
                                  use_openphish: bool = True,
                                  balance_classes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and combine all datasets with feature extraction.
        
        Args:
            use_phiusiil (bool): Include PhiUSIIL dataset
            use_phishtank (bool): Include PhishTank dataset
            use_mendeley (bool): Include Mendeley dataset
            balance_classes (bool): Balance legitimate/phishing classes
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Combined feature matrix X and labels y
        """
        print("=" * 70)
        print("MULTI-DATASET LOADING")
        print("=" * 70)
        
        all_dfs = []
        
        # Load Mendeley Dataset (High Priority - Real Data)
        if use_mendeley:
            try:
                df_mendeley = self.load_mendeley_dataset()
                if len(df_mendeley) > 0:
                    all_dfs.append(df_mendeley)
                    print(f"  [OK] Mendeley: {len(df_mendeley)} URLs")
            except Exception as e:
                print(f"  ✗ Mendeley failed: {e}")
        
        # Load PhiUSIIL dataset (primary source)
        if use_phiusiil:
            try:
                df_phiusiil = self.download_phiusiil_dataset()
                if len(df_phiusiil) > 0:
                    all_dfs.append(df_phiusiil)
                    print(f"  [OK] PhiUSIIL: {len(df_phiusiil)} URLs")
            except Exception as e:
                print(f"  ✗ PhiUSIIL failed: {e}")
        
        # Load OpenPhish dataset (live phishing feed)
        if use_openphish:
            try:
                df_openphish = self.download_openphish_dataset(limit=10000)
                if len(df_openphish) > 0:
                    all_dfs.append(df_openphish)
                    print(f"  [OK] OpenPhish: {len(df_openphish)} URLs (phishing)")
                    # Balance with legitimate URLs
                    df_legit_op = self.create_legitimate_urls(count=len(df_openphish))
                    all_dfs.append(df_legit_op)
                    print(f"  [OK] Legitimate (for OpenPhish balance): {len(df_legit_op)} URLs")
            except Exception as e:
                print(f"  ✗ OpenPhish failed: {e}")

        # Load PhishTank dataset
        if use_phishtank:
            try:
                df_phishtank = self.download_phishtank_dataset(limit=10000)
                if len(df_phishtank) > 0:
                    all_dfs.append(df_phishtank)
                    print(f"  [OK] PhishTank: {len(df_phishtank)} URLs (phishing)")
                    
                    # Add legitimate URLs to balance
                    df_legitimate = self.create_legitimate_urls(count=10000)
                    all_dfs.append(df_legitimate)
                    print(f"  [OK] Legitimate: {len(df_legitimate)} URLs")
            except Exception as e:
                print(f"  ✗ PhishTank failed: {e}")
        
        # If no datasets loaded, fall back to synthetic data
        if not all_dfs:
            print("\n[!] No datasets loaded. Using fallback strategy...")
            print("Creating balanced synthetic dataset for training...")
            df_legitimate = self.create_legitimate_urls(count=5000)
            all_dfs.append(df_legitimate)
        
        # Combine all datasets
        print(f"\n[*] Combining datasets...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates
        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['url'])
        if len(combined_df) < original_len:
            print(f"  Removed {original_len - len(combined_df)} duplicate URLs")
        
        # Balance classes if requested
        if balance_classes:
            legitimate_count = (combined_df['label'] == 0).sum()
            phishing_count = (combined_df['label'] == 1).sum()
            
            print(f"\n[*] Class distribution:")
            print(f"  Legitimate: {legitimate_count} ({legitimate_count/len(combined_df)*100:.1f}%)")
            print(f"  Phishing: {phishing_count} ({phishing_count/len(combined_df)*100:.1f}%)")
            
            # If imbalanced, downsample majority class
            if abs(legitimate_count - phishing_count) > 0.2 * max(legitimate_count, phishing_count):
                print(f"\n[*] Balancing classes...")
                min_count = min(legitimate_count, phishing_count)
                
                df_legitimate = combined_df[combined_df['label'] == 0].sample(n=min_count, random_state=42)
                df_phishing = combined_df[combined_df['label'] == 1].sample(n=min_count, random_state=42)
                
                combined_df = pd.concat([df_legitimate, df_phishing], ignore_index=True)
                combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
                
                print(f"  Balanced to {len(combined_df)} URLs ({min_count} per class)")
        
        # Save combined dataset
        combined_file = self.cache_dir / 'combined_dataset.csv'
        combined_df.to_csv(combined_file, index=False)
        print(f"\n[*] Saved combined dataset to {combined_file}")
        
        # Extract features
        print(f"\n" + "=" * 70)
        print("FEATURE EXTRACTION")
        print("=" * 70)
        
        X, y = self.extract_features_from_urls(combined_df)
        
        print(f"\n[SUCCESS] Dataset ready!")
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Legitimate: {(y == 0).sum()}")
        print(f"  Phishing: {(y == 1).sum()}")
        print("=" * 70 + "\n")
        
        return X, y


def main():
    """Test the multi-dataset loader."""
    loader = MultiDatasetLoader()
    X, y = loader.load_and_combine_datasets()
    
    print(f"\n[SUCCESS] Successfully loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")


if __name__ == '__main__':
    main()

