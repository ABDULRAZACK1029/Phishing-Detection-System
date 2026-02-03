"""
URL scanning routes for phishing detection
Integrates ML-based phishing detection with database storage
"""

from flask import Blueprint, request, jsonify
from models import ScanResult, db
from auth import token_required
from datetime import datetime
import uuid
import re
import json

# Import ML predictor
try:
    from ml.predict import predict_url
    ML_AVAILABLE = True
    print("[OK] ML predictor module loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️  ML predictor not available: {e}")
    print("   Using rule-based detection as fallback")

# Fallback detector if ML not available
if not ML_AVAILABLE:
    try:
        from ml.detector import PhishingDetector
        detector = PhishingDetector()
    except ImportError:
        detector = None
        print("⚠️  No detection method available")

scan_bp = Blueprint('scan', __name__)


def validate_url(url: str) -> tuple[bool, str]:
    """
    Validate and sanitize URL input.
    
    Args:
        url (str): URL to validate
        
    Returns:
        tuple: (is_valid, error_message or cleaned_url)
    """
    if not url or not isinstance(url, str):
        return False, "URL is required and must be a string"
    
    url = url.strip()
    
    # Check length limits
    if len(url) < 3:
        return False, "URL is too short"
    
    if len(url) > 2048:
        return False, "URL exceeds maximum length of 2048 characters"
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://', 'ftp://')):
        url = 'https://' + url
    
    # Basic URL pattern validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(url):
        return False, "Invalid URL format"
    
    return True, url


@scan_bp.route('/url', methods=['POST'])
@token_required
def scan_url(current_user):
    """
    Scan URL for phishing threats using ML model.
    
    Request JSON:
        {
            "url": "https://example.com"
        }
        
    Response JSON:
        {
            "scan_id": "ABC123XYZ",
            "is_safe": true/false,
            "threat_level": "safe"|"low"|"medium"|"high"|"critical",
            "threat_type": "Type of threat detected",
            "explanation": "Human-readable explanation",
            "scan_result": {Full scan result object}
        }
    """
    try:
        data = request.get_json()
        
        if not data or not data.get('url'):
            return jsonify({'error': 'URL is required'}), 400
        
        # Validate and sanitize URL
        is_valid, result = validate_url(data['url'])
        if not is_valid:
            return jsonify({'error': result}), 400
        
        url = result
        
        # Get IP address from request
        ip_address = request.remote_addr
        if request.headers.get('X-Forwarded-For'):
            ip_address = request.headers.get('X-Forwarded-For').split(',')[0].strip()
        
        # Perform ML-based phishing detection
        if ML_AVAILABLE:
            # Use ML predictor
            ml_result = predict_url(url)
            
            result = {
                'is_safe': ml_result['is_safe'],
                'threat_level': ml_result['threat_level'],
                'ml_score': ml_result['ml_score'],
                'features': ml_result['features']
            }
            
            # Determine threat type based on features and threat level
            threat_type = _determine_threat_type(ml_result)
            
            # Generate explanation
            explanation = _generate_explanation(url, ml_result)
            
        else:
            # Fallback to rule-based detection
            if detector:
                result = detector.detect(url)
                threat_type = result.get('threat_type', 'Unknown Threat')
                explanation = result.get('explanation', 'Detection performed using rule-based analysis')
            else:
                # Ultimate fallback
                result = {
                    'is_safe': True,
                    'threat_level': 'safe',
                    'ml_score': 0.0,
                    'features': {}
                }
                threat_type = None
                explanation = 'Detection service temporarily unavailable'
        
        # Generate unique scan ID
        scan_id = f"{uuid.uuid4().hex[:8].upper()}X"
        
        # Create scan result in database
        scan_result = ScanResult(
            user_id=current_user.id,
            url=url,
            is_safe=result['is_safe'],
            threat_level=result['threat_level'],
            threat_type=threat_type,
            ai_explanation=explanation,
            scan_id=scan_id,
            ip_address=ip_address,
            ml_score=result.get('ml_score'),
            ml_features=json.dumps(result.get('features', {})) if result.get('features') else None
        )
        
        db.session.add(scan_result)
        db.session.commit()
        
        # Update user's last active timestamp
        current_user.last_active = datetime.utcnow()
        db.session.commit()
        
        # Return response
        return jsonify({
            'scan_id': scan_id,
            'is_safe': result['is_safe'],
            'threat_level': result['threat_level'],
            'threat_type': threat_type,
            'explanation': explanation,
            'ml_confidence': result.get('ml_score', 0.0),
            'scan_result': scan_result.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Scan error: {e}")
        return jsonify({'error': 'An error occurred during scanning. Please try again.'}), 500


def _determine_threat_type(ml_result: dict) -> str:
    """
    Determine specific threat type based on ML results and features.
    
    Args:
        ml_result (dict): ML prediction result
        
    Returns:
        str: Threat type description
    """
    features = ml_result.get('features', {})
    threat_level = ml_result.get('threat_level', 'safe')
    
    if ml_result['is_safe']:
        return None
    
    # Prioritize based on specific indicators
    if features.get('has_homograph', 0) > 0:
        return 'Homograph Attack'
    elif features.get('has_ip', 0) > 0:
        return 'IP Address Spoofing'
    elif features.get('suspicious_keyword_count', 0) > 3:
        return 'Credential Harvesting'
    elif features.get('subdomain_count', 0) > 4:
        return 'Domain Spoofing'
    elif threat_level in ['high', 'critical']:
        return 'Suspicious Domain'
    elif threat_level == 'medium':
        return 'Potential Phishing'
    else:
        return 'Low Risk Threat'


def _generate_explanation(url: str, ml_result: dict) -> str:
    """
    Generate human-readable explanation of the detection result.
    
    Args:
        url (str): The URL that was analyzed
        ml_result (dict): ML prediction result
        
    Returns:
        str: Human-readable explanation
    """
    features = ml_result.get('features', {})
    threat_level = ml_result.get('threat_level', 'safe')
    confidence = ml_result.get('confidence', 0.0)
    is_safe = ml_result.get('is_safe', True)
    
    if is_safe:
        return f"This URL appears to be safe. Our ML model is {confidence:.1%} confident it is legitimate. No obvious phishing indicators detected."
    
    # Build explanation based on detected features
    reasons = []
    
    if features.get('has_homograph', 0) > 0:
        reasons.append("contains suspicious non-ASCII characters (possible homograph attack)")
    
    if features.get('has_ip', 0) > 0:
        reasons.append("uses an IP address instead of a domain name")
    
    suspicious_count = features.get('suspicious_keyword_count', 0)
    if suspicious_count > 0:
        reasons.append(f"contains {int(suspicious_count)} suspicious keywords (e.g., 'login', 'verify', 'update')")
    
    subdomain_count = features.get('subdomain_count', 0)
    if subdomain_count > 3:
        reasons.append(f"has an unusually high number of subdomains ({int(subdomain_count)})")
    
    if features.get('url_length', 0) > 100:
        reasons.append("has an unusually long URL")
    
    if not features.get('is_trusted_domain', 0):
        reasons.append("domain does not match known trusted domains")
    
    # Construct final explanation
    if reasons:
        reason_text = "; ".join(reasons)
        return f"⚠️ Warning: This URL {reason_text}. Threat level: {threat_level.upper()}. ML confidence: {confidence:.1%}."
    else:
        return f"⚠️ This URL has been flagged as potentially malicious with {confidence:.1%} confidence. Threat level: {threat_level.upper()}. Exercise caution."


@scan_bp.route('/history', methods=['GET'])
@token_required
def get_scan_history(current_user):
    """Get user's scan history with pagination."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Limit per_page to prevent abuse
        per_page = min(per_page, 100)
        
        scans = ScanResult.query.filter_by(user_id=current_user.id)\
            .order_by(ScanResult.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'scans': [scan.to_dict() for scan in scans.items],
            'total': scans.total,
            'page': page,
            'per_page': per_page,
            'pages': scans.pages
        }), 200
        
    except Exception as e:
        print(f"❌ History error: {e}")
        return jsonify({'error': 'Failed to retrieve scan history'}), 500

