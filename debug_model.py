
from ml.feature_extraction import FeatureExtractor
import numpy as np
import joblib

# Load model
model = joblib.load('ml/phishing_model.pkl')
extractor = FeatureExtractor()

# Typical college/university URLs
test_urls = [
    "https://portal.stanford.edu/student/login",
    "https://www.annauniv.edu/academic/courses/index.html",
    "http://erp.iitkgp.ac.in/SSOAdministration/login.htm",
    "https://www.srmist.edu.in/admission-international/",
    "https://my.college.edu.in/student-portal/exam-results?id=12345"
]

print("DEBUGGING FALSE POSITIVES\n")

for url in test_urls:
    print(f"URL: {url}")
    features = extractor.extract_features(url)
    
    # Predict
    vector = np.array([features[name] for name in extractor.get_feature_names()]).reshape(1, -1)
    prob = model.predict_proba(vector)[0]
    
    print(f"  Prediction: {'PHISHING' if prob[1] > 0.5 else 'LEGITIMATE'} ({prob[1]:.2%} phishing prob)")
    
    # Show suspicious triggers
    print("  Suspicious Triggers:")
    if features['suspicious_keyword_count'] > 0:
        print(f"    - Keywords: {features['suspicious_keyword_count']}")
    if features['entropy'] > 3.8:
        print(f"    - High Entropy: {features['entropy']:.2f}")
    if features['path_depth'] > 3:
        print(f"    - Deep Path: {features['path_depth']}")
    if features['num_dots'] > 3:
        print(f"    - Many Dots: {features['num_dots']}")
    if features['is_trusted_domain'] == 0:
        print(f"    - Not in trusted list")
        
    print("-" * 50)
