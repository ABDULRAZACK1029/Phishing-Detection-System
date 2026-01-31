# Phishing Detection Datasets

## Overview

This directory contains cached datasets used for training the phishing detection ML model.

## Datasets Included

### 1. PhiUSIIL Dataset (UCI ML Repository)
- **Source**: UCI Machine Learning Repository
- **Size**: 235,795 URLs (134,850 legitimate + 100,945 phishing)
- **File**: `phiusiil_dataset.csv`
- **Columns**: `url`, `label`
- **Download**: Automatic via `ucimlrepo` package

### 2. PhishTank Live Data
- **Source**: http://data.phishtank.com/
- **Size**: ~10,000 active phishing URLs (limited)
- **File**: `phishtank_dataset.csv`
- **Columns**: `url`, `label`
- **Note**: All entries are phishing URLs (label=1)

### 3. Legitimate URL Collection
- **Source**: Generated from trusted domains
- **Size**: ~10,000 URLs
- **File**: `legitimate_urls.csv`
- **Columns**: `url`, `label`
- **Purpose**: Balance PhishTank phishing-only dataset

### 4. Combined Dataset
- **File**: `combined_dataset.csv`
- **Description**: Merged dataset from all sources
- **Processing**: Deduplicated, balanced, shuffled

## File Format

All CSV files follow this format:

```csv
url,label
https://www.google.com,0
http://phishing-site-example.com,1
```

Where:
- `url`: The full URL string
- `label`: 0 = Legitimate, 1 = Phishing

## Caching

Datasets are downloaded once and cached locally to avoid repeated downloads:
- First run: Downloads from sources (~5-10 minutes)
- Subsequent runs: Loads from cache (~1 second)

## Installation

To download the PhiUSIIL dataset automatically, install:

```bash
pip install ucimlrepo
```

## Manual Download

If automatic download fails, you can manually download:

1. **PhiUSIIL**: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
2. **PhishTank**: http://data.phishtank.com/data/online-valid.csv

Place files in this directory with names matching the cache files.

## Generated Files

- `phiusiil_dataset.csv` - PhiUSIIL data
- `phishtank_dataset.csv` - PhishTank data  
- `phishtank_raw.csv` - Raw PhishTank download (temporary)
- `legitimate_urls.csv` - Generated legitimate URLs
- `combined_dataset.csv` - Final merged dataset

## Dataset Statistics

After loading and processing, expect:
- Total URLs: ~20,000-250,000 (depending on sources)
- Class balance: ~50% legitimate, ~50% phishing
- Features extracted: 17 per URL
