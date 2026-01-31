# PhishGuard - Full-Stack Web Application

A full-stack phishing detection web application with Flask backend, SQLite database, JWT authentication, and ML-ready architecture.

## Features

- **User Authentication**: JWT-based authentication with password hashing
- **Role-Based Access**: User and Admin roles with different permissions
- **URL Phishing Detection**: AI-powered URL analysis with threat scoring
- **Admin Dashboard**: Analytics, user management, and threat logs
- **ML-Ready Architecture**: Modular design for scikit-learn integration
- **RESTful API**: Clean API endpoints for all operations

## Tech Stack

### Backend
- Python Flask
- SQLAlchemy (SQLite)
- JWT (PyJWT)
- bcrypt for password hashing
- scikit-learn (for ML integration)

### Frontend
- HTML5/CSS3
- Tailwind CSS
- JavaScript (Vanilla)
- Chart.js for analytics

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Initialize the database:**
The database will be automatically created when you run the application.

3. **Run the Flask server:**
```bash
python app.py
```

The server will start on `http://localhost:5000`

## Default Credentials

- **Admin Account:**
  - Email: `admin@phishguard.com`
  - Password: `admin123`

## API Endpoints

### Authentication (`/api/auth`)
- `POST /register` - User registration
- `POST /login` - User login
- `GET /verify` - Verify JWT token

### Scanning (`/api/scan`)
- `POST /url` - Scan URL for phishing (requires auth)
- `GET /history` - Get user's scan history (requires auth)

### Admin (`/api/admin`)
- `GET /dashboard/stats` - Get dashboard statistics (admin only)
- `GET /users` - Get all users (admin only)
- `POST /users` - Create new user (admin only)
- `PUT /users/<id>` - Update user (admin only)
- `DELETE /users/<id>` - Delete user (admin only)
- `GET /threats` - Get threat logs (admin only)
- `GET /analytics/threat-types` - Get threat type analytics (admin only)
- `GET /analytics/user-activity` - Get user activity analytics (admin only)

### User (`/api/user`)
- `GET /profile` - Get user profile (requires auth)
- `PUT /profile` - Update user profile (requires auth)

## Project Structure

```
.
├── app.py                 # Main Flask application
├── models.py              # SQLAlchemy database models
├── auth.py                # JWT authentication utilities
├── requirements.txt       # Python dependencies
├── routes/                # API route blueprints
│   ├── auth.py           # Authentication routes
│   ├── scan.py           # URL scanning routes
│   ├── admin.py          # Admin routes
│   └── user.py           # User routes
├── ml/                    # ML module for phishing detection
│   ├── detector.py       # Phishing detection logic
│   └── trainer.py        # ML model trainer (example)
├── login.html            # Login page
├── register.html         # Registration page
├── user_dashboard.html   # User dashboard
├── admin_dashboard.html  # Admin dashboard
├── users.html            # User management page
└── threat_logs.html      # Threat logs page
```

## ML Integration

The system is designed to be easily extended with scikit-learn models:

1. **Feature Extraction**: `ml/detector.py` extracts features from URLs
2. **Model Training**: `ml/trainer.py` provides example training code
3. **Prediction**: The detector can use trained models via `predict_with_model()`

To integrate a trained model:
1. Train your model using `ml/trainer.py` or your own training script
2. Save the model to `ml/phishing_model.pkl`
3. Load the model in `ml/detector.py` and use `predict_with_model()`

## Database Schema

### Users Table
- `id` (Primary Key)
- `email` (Unique)
- `password_hash`
- `name`
- `role` ('user' or 'admin')
- `created_at`
- `last_active`
- `is_active`

### Scan Results Table
- `id` (Primary Key)
- `user_id` (Foreign Key)
- `url`
- `is_safe` (Boolean)
- `threat_level` ('safe', 'low', 'medium', 'high', 'critical')
- `threat_type`
- `ai_explanation`
- `scan_id` (Unique)
- `ip_address`
- `created_at`
- `ml_score`
- `ml_features` (JSON)

## Development

### Running in Development Mode
```bash
python app.py
```

The app runs in debug mode by default. Change `debug=True` to `False` in production.

### Database Reset
Delete `phishguard.db` to reset the database. The default admin user will be recreated on next startup.

## Security Notes

- Change `SECRET_KEY` in production
- Use environment variables for sensitive configuration
- Implement rate limiting for API endpoints
- Add HTTPS in production
- Regularly update dependencies

## License

ISC
