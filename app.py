"""
PhishGuard - Full-Stack Web Application
Main Flask application entry point
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import timedelta

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///phishguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
from models import db
db.init_app(app)
CORS(app, supports_credentials=True)

# Import models (must be after db initialization)
from models import User, ScanResult

# Import routes
from routes.auth import auth_bp
from routes.scan import scan_bp
from routes.admin import admin_bp
from routes.user import user_bp

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(scan_bp, url_prefix='/api/scan')
app.register_blueprint(admin_bp, url_prefix='/api/admin')
app.register_blueprint(user_bp, url_prefix='/api/user')

# Serve static HTML files
@app.route('/')
def index():
    return send_from_directory('.', 'login.html')

@app.route('/<path:path>')
def serve_static(path):
    if path.endswith('.html'):
        return send_from_directory('.', path)
    return send_from_directory('.', path)

# Initialize database
def init_db():
    """Initialize database and create default admin user"""
    db.create_all()
    # Create default admin user if not exists
    admin = User.query.filter_by(email='admin@phishguard.com').first()
    if not admin:
        from werkzeug.security import generate_password_hash
        admin = User(
            email='admin@phishguard.com',
            password_hash=generate_password_hash('admin123'),
            name='Admin User',
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
