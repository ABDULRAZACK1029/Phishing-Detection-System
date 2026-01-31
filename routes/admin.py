"""
Admin routes for dashboard analytics and user management
"""

from flask import Blueprint, request, jsonify
from models import User, ScanResult, db
from auth import admin_required
from datetime import datetime, timedelta
from sqlalchemy import func, and_

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/dashboard/stats', methods=['GET'])
@admin_required
def get_dashboard_stats(current_user):
    """Get admin dashboard statistics"""
    try:
        # Total users
        total_users = User.query.count()
        
        # Active users (active in last 24 hours)
        active_users = User.query.filter(
            User.last_active >= datetime.utcnow() - timedelta(days=1)
        ).count()
        
        # Admin users
        admin_users = User.query.filter_by(role='admin').count()
        
        # Total scans
        total_scans = ScanResult.query.count()
        
        # Average scans per user
        avg_scans = db.session.query(func.avg(
            db.session.query(func.count(ScanResult.id))
            .filter(ScanResult.user_id == User.id)
            .scalar_subquery()
        )).scalar() or 0
        
        # Recent scans (last 24 hours)
        recent_scans = ScanResult.query.filter(
            ScanResult.created_at >= datetime.utcnow() - timedelta(days=1)
        ).count()
        
        # Threat statistics
        threat_stats = db.session.query(
            ScanResult.threat_level,
            func.count(ScanResult.id)
        ).group_by(ScanResult.threat_level).all()
        
        threat_breakdown = {level: count for level, count in threat_stats}
        
        return jsonify({
            'total_users': total_users,
            'active_users': active_users,
            'admin_users': admin_users,
            'total_scans': total_scans,
            'avg_scans_per_user': round(float(avg_scans), 1),
            'recent_scans': recent_scans,
            'threat_breakdown': threat_breakdown
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users', methods=['GET'])
@admin_required
def get_users(current_user):
    """Get all users with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        search = request.args.get('search', '').strip()
        role_filter = request.args.get('role', 'all')
        
        query = User.query
        
        # Apply search filter
        if search:
            query = query.filter(
                db.or_(
                    User.email.ilike(f'%{search}%'),
                    User.name.ilike(f'%{search}%')
                )
            )
        
        # Apply role filter
        if role_filter != 'all':
            query = query.filter_by(role=role_filter)
        
        users = query.order_by(User.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'users': [user.to_dict() for user in users.items],
            'total': users.total,
            'page': page,
            'per_page': per_page,
            'pages': users.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users/<int:user_id>', methods=['GET'])
@admin_required
def get_user_details(current_user, user_id):
    """Get user details"""
    try:
        user = User.query.get_or_404(user_id)
        return jsonify({'user': user.to_dict()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users', methods=['POST'])
@admin_required
def create_user(current_user):
    """Create new user (admin only)"""
    try:
        data = request.get_json()
        
        if not data or not data.get('email') or not data.get('password') or not data.get('name'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        email = data['email'].strip().lower()
        password = data['password']
        name = data['name'].strip()
        role = data.get('role', 'user')
        
        # Validate role
        if role not in ['user', 'admin']:
            return jsonify({'error': 'Invalid role'}), 400
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'User already exists'}), 409
        
        # Create user
        user = User(
            email=email,
            name=name,
            role=role
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'User created successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(current_user, user_id):
    """Update user"""
    try:
        user = User.query.get_or_404(user_id)
        data = request.get_json()
        
        if 'name' in data:
            user.name = data['name'].strip()
        if 'role' in data:
            if data['role'] not in ['user', 'admin']:
                return jsonify({'error': 'Invalid role'}), 400
            user.role = data['role']
        if 'is_active' in data:
            user.is_active = bool(data['is_active'])
        if 'password' in data and data['password']:
            user.set_password(data['password'])
        
        db.session.commit()
        
        return jsonify({
            'message': 'User updated successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(current_user, user_id):
    """Delete user"""
    try:
        user = User.query.get_or_404(user_id)
        
        # Prevent deleting yourself
        if user.id == current_user.id:
            return jsonify({'error': 'Cannot delete your own account'}), 400
        
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'message': 'User deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/threats', methods=['GET'])
@admin_required
def get_threat_logs(current_user):
    """Get threat logs (scan results)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        severity_filter = request.args.get('severity', 'all')
        search = request.args.get('search', '').strip()
        
        query = ScanResult.query.filter(ScanResult.is_safe == False)
        
        # Apply severity filter
        if severity_filter != 'all':
            query = query.filter_by(threat_level=severity_filter)
        
        # Apply search filter
        if search:
            query = query.filter(
                db.or_(
                    ScanResult.url.ilike(f'%{search}%'),
                    ScanResult.threat_type.ilike(f'%{search}%')
                )
            )
        
        threats = query.order_by(ScanResult.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        # Include user information
        results = []
        for threat in threats.items:
            result = threat.to_dict()
            result['user'] = threat.user.to_dict() if threat.user else None
            results.append(result)
        
        return jsonify({
            'threats': results,
            'total': threats.total,
            'page': page,
            'per_page': per_page,
            'pages': threats.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/analytics/threat-types', methods=['GET'])
@admin_required
def get_threat_type_analytics(current_user):
    """Get threat type distribution"""
    try:
        threat_types = db.session.query(
            ScanResult.threat_type,
            func.count(ScanResult.id)
        ).filter(ScanResult.is_safe == False)\
         .group_by(ScanResult.threat_type)\
         .all()
        
        return jsonify({
            'threat_types': {type_name: count for type_name, count in threat_types if type_name}
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/analytics/user-activity', methods=['GET'])
@admin_required
def get_user_activity_analytics(current_user):
    """Get user activity analytics"""
    try:
        # Users with most threats detected
        top_users = db.session.query(
            User.id,
            User.email,
            User.name,
            func.count(ScanResult.id).label('threat_count')
        ).join(ScanResult, User.id == ScanResult.user_id)\
         .filter(ScanResult.is_safe == False)\
         .group_by(User.id)\
         .order_by(func.count(ScanResult.id).desc())\
         .limit(10)\
         .all()
        
        return jsonify({
            'top_users': [
                {
                    'user_id': user_id,
                    'email': email,
                    'name': name,
                    'threat_count': threat_count
                }
                for user_id, email, name, threat_count in top_users
            ]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
