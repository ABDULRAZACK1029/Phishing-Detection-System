"""
Database Models for PhishGuard
SQLAlchemy models for User and ScanResult
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize db here, will be initialized with app in app.py
db = SQLAlchemy()

class User(db.Model):
    """User model with authentication and role management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')  # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship to scan results
    scan_results = db.relationship('ScanResult', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'is_active': self.is_active,
            'scan_count': len(self.scan_results)
        }
    
    def __repr__(self):
        return f'<User {self.email}>'


class ScanResult(db.Model):
    """Scan result model for storing URL phishing check results"""
    __tablename__ = 'scan_results'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    url = db.Column(db.Text, nullable=False)
    is_safe = db.Column(db.Boolean, nullable=False)
    threat_level = db.Column(db.String(20), nullable=False)  # 'safe', 'low', 'medium', 'high', 'critical'
    threat_type = db.Column(db.String(100), nullable=True)  # e.g., 'Credential Harvesting', 'Domain Spoofing'
    ai_explanation = db.Column(db.Text, nullable=True)
    scan_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    ip_address = db.Column(db.String(45), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # ML features (for future ML integration)
    ml_score = db.Column(db.Float, nullable=True)
    ml_features = db.Column(db.Text, nullable=True)  # JSON string of features
    
    def to_dict(self):
        """Convert scan result to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'url': self.url,
            'is_safe': self.is_safe,
            'threat_level': self.threat_level,
            'threat_type': self.threat_type,
            'ai_explanation': self.ai_explanation,
            'scan_id': self.scan_id,
            'ip_address': self.ip_address,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'ml_score': self.ml_score
        }
    
    def __repr__(self):
        return f'<ScanResult {self.scan_id}>'
