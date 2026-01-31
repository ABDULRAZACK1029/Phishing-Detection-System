# Machine Learning Module - Phishing Detection

## Overview

This ML module implements a **Random Forest Classifier** for detecting phishing websites based on URL features. The system is designed for academic use in a BCA Final Year project and follows best practices for explainability and modularity.

## Architecture

```
ml/
├── feature_extraction.py    # Extract 17 URL-based features
├── trainer.py               # Train Random Forest/Logistic Regression
├── predict.py               # Prediction API with thread-safe model loading
├── detector.py              # Legacy rule-based detector (fallback)
├── phishing_model.pkl       # Trained model (generated after training)
├── model_metadata.json      # Model performance metrics
└── README.md                # This file
```

## Features Extracted

The system analyzes URLs using **17 features**:

### Length-Based Features
1. **URL Length** - Total character count
2. **Domain Length** - Domain name length
3. **Path Length** - URL path length

### Character Count Features
4. **Number of Dots (.)** - Subdomain indicators
5. **Number of Hyphens (-)** - Domain obfuscation
6. **Number of Slashes (/)** - Path complexity
7. **Number of Question Marks (?)** - Query parameters
8. **Number of Equals (=)** - Parameter assignments
9. **Number of Ampersands (&)** - Multiple parameters
10. **Number of Percent (%)** - URL encoding usage

### Binary Features
11. **Has IP Address** - Using IP instead of domain
12. **Has Port Number** - Non-standard port usage
13. **Is Trusted Domain** - Whitelist check
14. **Has Homograph Characters** - Unicode spoofing detection

### Count Features
15. **Suspicious Keyword Count** - Common phishing terms (login, verify, update, bank, etc.)
16. **Subdomain Count** - Number of subdomains
17. **Path Depth** - Directory nesting level

## ML Model

### Algorithm: Random Forest Classifier

**Why Random Forest?**
- ✅ Excellent interpretability for academic defense
- ✅ Built-in feature importance metrics
- ✅ Robust to overfitting with proper configuration
- ✅ No feature scaling required
- ✅ Handles non-linear relationships well

**Model Configuration:**
```python
RandomForestClassifier(
    n_estimators=200,         # 200 decision trees
    max_depth=15,             # Maximum tree depth
    min_samples_split=5,      # Minimum samples to split node
    min_samples_leaf=2,       # Minimum samples in leaf
    random_state=42           # Reproducibility
)
```

### Alternative: Logistic Regression

For simpler explanations or comparison:
```python
LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)
```

## Usage

### 1. Training the Model

```bash
# Navigate to project root
cd C:\Users\Dell\Desktop\frontend

# Train the model
python ml/trainer.py
```

This will:
- Generate 5000 synthetic training samples
- Split data 80/20 (train/test)
- Train Random Forest model
- Display accuracy, confusion matrix, classification report
- Save model to `ml/phishing_model.pkl`
- Save metadata to `ml/model_metadata.json`

**Expected Output:**
```
======================================================================
PHISHING DETECTION MODEL TRAINING
======================================================================

Generating 5000 synthetic samples...
✓ Generated 5000 samples (Legitimate: 2500, Phishing: 2500)

Dataset split:
  Training samples: 4000
  Testing samples: 1000
  Features: 17

📊 Training Random Forest Classifier...
✓ Training completed

----------------------------------------------------------------------
MODEL EVALUATION
----------------------------------------------------------------------

🎯 Overall Accuracy: 95.00%

📊 Confusion Matrix:
                  Predicted
                Legit  Phishing
Actual  Legit     480      20
        Phishing   30     470

📈 Classification Report:
              precision    recall  f1-score   support
  Legitimate       0.94      0.96      0.95       500
    Phishing       0.96      0.94      0.95       500

🔍 Top 10 Most Important Features:
    1. suspicious_keyword_count    - 0.2150
    2. has_ip                      - 0.1820
    3. subdomain_count             - 0.1340
    ...
```

### 2. Making Predictions

```python
from ml.predict import predict_url

# Predict a URL
result = predict_url('https://secure-bank-verify.com/login')

print(result['prediction'])    # 'Phishing' or 'Legitimate'
print(result['confidence'])    # 0.0 to 1.0
print(result['threat_level'])  # 'safe', 'low', 'medium', 'high', 'critical'
```

### 3. Flask Integration

The model is automatically loaded when Flask starts:

```python
# In routes/scan.py
from ml.predict import predict_url

# API endpoint handles prediction
POST /api/scan/url
{
    "url": "https://example.com"
}
```

## Upgrading with Real Data

### Step 1: Download Phishing Dataset

**Option A: PhishTank**
```bash
# Download from https://www.phishtank.com/developer_info.php
# Requires free API key
```

**Option B: UCI ML Repository**
```bash
# Download from:
# https://archive.ics.uci.edu/ml/datasets/phishing+websites
```

**Option C: Kaggle**
```bash
# Search for "phishing URL dataset" on Kaggle
```

### Step 2: Prepare CSV File

Create a CSV with columns matching feature names:
```csv
url_length,domain_length,path_length,...,label
65,15,20,...,0
180,35,90,...,1
```

Where `label` is:
- `0` = Legitimate
- `1` = Phishing

### Step 3: Retrain Model

```python
from ml.trainer import PhishingModelTrainer

trainer = PhishingModelTrainer()

# Load real data
X, y = trainer.load_real_data('path/to/dataset.csv')

# Train with real data
trainer.train(X=X, y=y, model_type='random_forest')

# Save model
trainer.save_model()
```

## Future Upgrades: Deep Learning

The modular design allows easy integration of deep learning:

### Option 1: LSTM for Sequential Analysis
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# URL character sequence model
model = Sequential([
    Embedding(input_dim=128, output_dim=64),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

### Option 2: BERT for NLP-Based Detection
```python
from transformers import BertTokenizer, BertForSequenceClassification

# Pre-trained BERT for URL classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

### Option 3: Ensemble Approach
Combine multiple models for better accuracy:
```python
# Random Forest + LSTM + BERT ensemble
final_prediction = (0.4 * rf_pred + 0.3 * lstm_pred + 0.3 * bert_pred)
```

## Performance Optimization

### Model Loading
- Model is lazy-loaded on first prediction (saves startup time)
- Thread-safe singleton pattern prevents multiple loads
- Cached in memory for subsequent predictions

### Prediction Speed
- Feature extraction: ~1ms per URL
- Model prediction: ~5-10ms per URL (Random Forest)
- Total latency: <20ms per request

## Academic Explanation Guide

### For Viva Defense

**Q: Why Random Forest over other algorithms?**
> Random Forest provides excellent interpretability through feature importance, which helps explain WHY a URL is phishing. It's also robust to overfitting and doesn't require feature scaling.

**Q: How do you handle class imbalance?**
> We use stratified train-test split to maintain equal representation of both classes (50% legitimate, 50% phishing) in training and testing.

**Q: What is the confusion matrix telling us?**
> The confusion matrix shows True Positives (correctly detected phishing), True Negatives (correctly identified legitimate), False Positives (legitimate flagged as phishing), and False Negatives (phishing missed).

**Q: How do you prevent overfitting?**
> We limit tree depth to 15, require minimum 5 samples to split nodes, and use 200 trees with random feature selection to ensure generalization.

**Q: Can this be upgraded to deep learning?**
> Yes, the modular design allows replacing the Random Forest with LSTM or BERT models for sequential/NLP-based analysis while keeping the same prediction API.

## Troubleshooting

### Model file not found
```bash
# Make sure to train the model first
python ml/trainer.py
```

### Import errors
```bash
# Install dependencies
pip install -r requirements.txt
```

### Low accuracy
- Increase training samples: `trainer.create_sample_data(n_samples=10000)`
- Try different model: `trainer.train(model_type='logistic_regression')`
- Use real phishing dataset instead of synthetic data

## References

1. **Phishing Detection Research:**
   - "Phishing Websites Detection based on Machine Learning" (2015)
   - UCI ML Phishing Dataset Documentation

2. **Random Forest:**
   - Breiman, L. (2001). "Random Forests". Machine Learning

3. **Phishing Datasets:**
   - PhishTank: https://www.phishtank.com/
   - UCI Repository: https://archive.ics.uci.edu/ml/

## License

Academic use only - BCA Final Year Project
