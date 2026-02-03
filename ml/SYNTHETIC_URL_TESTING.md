# Synthetic Phishing URL Testing

## Overview

This module provides **realistic synthetic phishing URL generation and testing** for the PhishGuard phishing detection system. It enables ethical, legal testing of the ML model using programmatically generated phishing-style URLs without relying on real malicious websites.

> [!IMPORTANT]
> **Academic & Ethical Use Only**: This feature is designed for cybersecurity education and research in an academic BCA Final Year project. All synthetic URLs use **safe TLDs** (.test, .example, .invalid) as defined in RFC 2606 and RFC 6761, ensuring no real domains are affected.

## Features

✅ **Realistic Attacker Patterns**
- Brand impersonation (banking, ecommerce, social media, cloud services)
- Typosquatting (substitution, omission, insertion, transposition, repetition)
- Suspicious subdomain manipulation
- Hyphenated security keywords
- HTTPS misuse and port manipulation

✅ **Safe & Ethical**
- Uses only RFC-reserved TLDs (.test, .example, .invalid)
- No real websites affected
- No legal issues with domain squatting
- Compliant with academic integrity standards

✅ **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- False Positive/Negative Rate
- ROC-AUC Score
- Error analysis with misclassified URLs

✅ **Seamless Integration**
- Works with existing `FeatureExtractor` pipeline
- Compatible with trained scikit-learn models
- No breaking changes to existing functionality

## Quick Start

### 1. Generate and Test Synthetic URLs

```bash
# Basic usage - generate and test 1000 URLs
python ml/test_synthetic_urls.py --count 1000

# Test specific categories
python ml/test_synthetic_urls.py --categories banking ecommerce --count 500

# High realism mode
python ml/test_synthetic_urls.py --realism high --count 2000

# Export results
python ml/test_synthetic_urls.py --count 1000 --export results.json

# Show error analysis
python ml/test_synthetic_urls.py --count 500 --analyze-errors --max-errors 20
```

### 2. Use in Python Code

```python
from ml.synthetic_url_generator import generate_synthetic_dataset
from ml.synthetic_url_evaluator import evaluate_synthetic_urls

# Generate 1000 synthetic URLs (500 phishing, 500 legitimate)
urls, labels = generate_synthetic_dataset(count=1000, realism="medium")

# Evaluate model performance
results = evaluate_synthetic_urls(urls, labels, export_path="results.json")

print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
print(f"F1-Score: {results['metrics']['f1_score']:.2%}")
```

## Modules

### 1. `synthetic_url_generator.py`

Generates realistic phishing-style URLs using attacker patterns.

**Key Classes**:
- `SyntheticURLGenerator`: Main generator with configurable patterns
- `BrandDatabase`: Stores legitimate brand names by category
- `AttackerPatterns`: Implements typosquatting techniques

**Example**:
```python
from ml.synthetic_url_generator import SyntheticURLGenerator, RealismLevel

generator = SyntheticURLGenerator(seed=42)

# Generate phishing URLs
phishing_urls = generator.generate_phishing_urls(
    count=100,
    realism=RealismLevel.HIGH
)

# Generate legitimate URLs
legit_urls = generator.generate_legitimate_urls(count=100)

# Generate balanced dataset
urls, labels = generator.generate_balanced_dataset(total_count=200)
```

### 2. `synthetic_url_evaluator.py`

Evaluates model performance on synthetic URLs with comprehensive metrics.

**Key Classes**:
- `SyntheticURLEvaluator`: Main evaluation engine

**Example**:
```python
from ml.synthetic_url_evaluator import SyntheticURLEvaluator

evaluator = SyntheticURLEvaluator()

# Evaluate model
results = evaluator.evaluate_urls(urls, labels)

# Show detailed results
evaluator.print_results()

# Analyze errors
evaluator.analyze_errors(max_display=10)

# Export results
evaluator.export_results("results.json", format="json")
```

### 3. `test_synthetic_urls.py`

Command-line interface for synthetic URL testing.

**Usage**:
```bash
python ml/test_synthetic_urls.py [OPTIONS]

Options:
  --count, -c          Total URLs to generate (default: 1000)
  --realism, -r        Realism level: low, medium, high (default: medium)
  --categories         Brand categories: banking, ecommerce, social, cloud, tech
  --seed               Random seed for reproducibility
  --model-path         Path to trained model
  --analyze-errors     Show detailed error analysis
  --max-errors         Max errors to display (default: 10)
  --export             Export results to file (JSON/CSV)
  --include-urls       Include full URL list in export
  --quiet, -q          Minimal output
  --show-samples       Show N sample URLs before evaluation
```

## Attacker Patterns

### Typosquatting Techniques

| Technique | Example | Description |
|-----------|---------|-------------|
| **Substitution** | `paypal` → `paypa1` | Replace characters with similar-looking ones (l→1, o→0) |
| **Omission** | `facebook` → `facbook` | Remove a character |
| **Insertion** | `amazon` → `amazzon` | Add an extra character |
| **Transposition** | `microsoft` → `micrsooft` | Swap adjacent characters |
| **Repetition** | `google` → `gooogle` | Repeat a character |

### Subdomain Manipulation

```
# Multi-level subdomains
secure.login.verify.paypal.test

# Brand in subdomain
paypal-secure.malicious.example

# Urgency keywords
urgent-verify.account.invalid
```

### Hyphenated Keywords

```
secure-login-verify.test
account-update-required.example
banking-security-alert.invalid
```

## Realism Levels

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **Low** | Obvious phishing | Multiple red flags, easy to detect, multiple typos |
| **Medium** | Moderate sophistication | Some obfuscation, realistic structure, 1-2 techniques |
| **High** | Advanced phishing | Subtle patterns, professional appearance, minimal typos |

## Brand Categories

- **Banking**: PayPal, Chase, Bank of America, Wells Fargo
- **Ecommerce**: Amazon, eBay, Alibaba, Walmart
- **Social**: Facebook, Twitter, Instagram, LinkedIn
- **Cloud**: Google, Microsoft, Dropbox, iCloud
- **Tech**: Apple, Adobe, Oracle, Samsung

## Evaluation Metrics

### Primary Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of URLs flagged as phishing, how many are actually phishing
- **Recall**: Of actual phishing URLs, how many were detected
- **F1-Score**: Harmonic mean of precision and recall

### Additional Metrics

- **Specificity**: True Negative Rate (correctly identified legitimate URLs)
- **False Positive Rate**: Legitimate URLs incorrectly flagged as phishing
- **False Negative Rate**: Phishing URLs missed by the model
- **ROC-AUC**: Area under the ROC curve

### Confusion Matrix

```
                  Predicted
               Legit  Phishing
Actual Legit     TN      FP
       Phish     FN      TP
```

- **TP (True Positives)**: Correctly identified phishing URLs
- **TN (True Negatives)**: Correctly identified legitimate URLs
- **FP (False Positives)**: Legitimate URLs flagged as phishing
- **FN (False Negatives)**: Phishing URLs missed by model

## Example Workflow

### Complete Testing Workflow

```bash
# Step 1: Train model (if not already trained)
python ml/improved_trainer.py

# Step 2: Generate and test 2000 synthetic URLs
python ml/test_synthetic_urls.py --count 2000 --realism medium

# Step 3: Test specific categories with high realism
python ml/test_synthetic_urls.py \
  --categories banking social \
  --count 1000 \
  --realism high \
  --analyze-errors \
  --export banking_social_results.json

# Step 4: Compare different realism levels
python ml/test_synthetic_urls.py --count 500 --realism low --export low_realism.json
python ml/test_synthetic_urls.py --count 500 --realism medium --export medium_realism.json
python ml/test_synthetic_urls.py --count 500 --realism high --export high_realism.json
```

### Python Integration Example

```python
from ml.synthetic_url_generator import SyntheticURLGenerator, RealismLevel, BrandCategory
from ml.synthetic_url_evaluator import SyntheticURLEvaluator

# Initialize generator
generator = SyntheticURLGenerator(seed=42)

# Generate URLs for banking category
urls, labels = generator.generate_balanced_dataset(
    total_count=500,
    categories=[BrandCategory.BANKING],
    realism=RealismLevel.HIGH
)

# Evaluate model
evaluator = SyntheticURLEvaluator()
results = evaluator.evaluate_urls(urls, labels)

# Check performance
if results['metrics']['accuracy'] >= 0.90:
    print("✓ Model performs well on banking phishing URLs")
else:
    print("⚠ Model needs improvement for banking phishing detection")

# Analyze errors
misclassified = evaluator.get_misclassified_urls()
print(f"False Positives: {len(misclassified['false_positives'])}")
print(f"False Negatives: {len(misclassified['false_negatives'])}")
```

## Interpreting Results

### Performance Benchmarks

| Metric | Excellent | Good | Moderate | Poor |
|--------|-----------|------|----------|------|
| Accuracy | ≥95% | 85-95% | 75-85% | <75% |
| Precision | ≥90% | 80-90% | 70-80% | <70% |
| Recall | ≥90% | 80-90% | 70-80% | <70% |
| F1-Score | ≥90% | 80-90% | 70-80% | <70% |

### Warning Signs

⚠️ **High False Positive Rate (>10%)**
- Many legitimate URLs incorrectly flagged as phishing
- May frustrate users with false alarms
- Consider adjusting model threshold or retraining

⚠️ **High False Negative Rate (>10%)**
- Many phishing URLs missed by the model
- Security risk - real phishing attacks may go undetected
- Model needs improvement on phishing pattern detection

## Ethical & Legal Compliance

> [!CAUTION]
> **Safe TLDs Only**: This implementation uses RFC 2606/6761 reserved TLDs (.test, .example, .invalid) that are **guaranteed never to be registered** as real domains.

**Why This Matters**:
- ✅ No real websites are affected
- ✅ No legal issues with domain squatting or trademark infringement
- ✅ Ethical research practices
- ✅ Academic integrity maintained
- ✅ Safe for educational demonstrations

**Documentation Statement for Academic Use**:
> "All synthetic URLs in this project are generated for educational and research purposes in an academic cybersecurity project. The URLs use reserved TLDs (.test, .example, .invalid) as specified in RFC 2606 and RFC 6761, ensuring no real domains are affected. No actual phishing websites are created, accessed, or distributed."

## Troubleshooting

### Model Not Found Error

```
❌ ERROR: Model file not found: ml/phishing_model.pkl
```

**Solution**: Train the model first
```bash
python ml/improved_trainer.py
```

### Import Errors

```
❌ ERROR: No module named 'synthetic_url_generator'
```

**Solution**: Run from project root directory
```bash
cd c:\Users\Dell\Desktop\frontend
python ml/test_synthetic_urls.py --count 1000
```

### Low Accuracy on Synthetic URLs

If the model performs poorly on synthetic URLs:

1. **Check Realism Level**: Start with `low` realism to verify basic detection
2. **Analyze Errors**: Use `--analyze-errors` to see which patterns are missed
3. **Retrain Model**: Consider retraining with more diverse data
4. **Feature Engineering**: Check if new attacker patterns need new features

## Best Practices

### For Academic Projects

1. **Document Everything**: Keep records of all test runs and results
2. **Use Multiple Realism Levels**: Test with low, medium, and high realism
3. **Test All Categories**: Ensure model works across different brand types
4. **Analyze Errors**: Understand why the model fails on certain patterns
5. **Compare Baselines**: Track improvements over time

### For Model Development

1. **Start Simple**: Begin with low realism to verify basic functionality
2. **Iterate**: Gradually increase realism to find model weaknesses
3. **Focus on Errors**: Prioritize fixing high false negative rates
4. **Balance Classes**: Ensure equal phishing/legitimate URL counts
5. **Use Seeds**: Set random seeds for reproducible experiments

## Integration with Existing System

This feature integrates seamlessly with the existing PhishGuard system:

- ✅ Uses existing `FeatureExtractor` for feature extraction
- ✅ Compatible with existing trained models
- ✅ No changes required to `app.py`, `routes/`, or `models.py`
- ✅ Standalone modules that don't affect production code
- ✅ Can be used for continuous model validation

## Future Enhancements

Potential improvements for future versions:

- [ ] Add more brand categories (cryptocurrency, gaming, streaming)
- [ ] Implement punycode/IDN homograph attacks
- [ ] Add URL shortener simulation
- [ ] Generate synthetic email phishing content
- [ ] Create visual similarity metrics for brand impersonation
- [ ] Add time-based analysis (model performance over time)
- [ ] Implement adversarial testing (generate URLs to fool the model)

## References

- **RFC 2606**: Reserved Top Level DNS Names (.test, .example)
- **RFC 6761**: Special-Use Domain Names (.invalid)
- **PhishTank**: Real-world phishing URL database
- **APWG**: Anti-Phishing Working Group research

## Support

For issues or questions about synthetic URL testing:

1. Check this documentation
2. Review example code in module docstrings
3. Run demo scripts: `python ml/synthetic_url_generator.py`
4. Check error messages and troubleshooting section

---

**Last Updated**: 2026-02-03  
**Version**: 1.0.0  
**Author**: BCA Final Year Project - PhishGuard Team
