# PhishGuard - Advanced Phishing Detection System

A comprehensive full-stack web application that leverages sophisticated Machine Learning models to detect phishing websites in real-time. Built with a Flask backend and a modern vanilla Javascript frontend, it features a robust Multi-Model Ensemble system for high accuracy.

## 🚀 Key Features

### Machine Learning Engine
- **Multi-Model Ensemble**: Combines Random Forest and Gradient Boosting models for superior accuracy (>95%).
- **Advanced Feature Extraction**: Analyzes over 30 URL features including lexical patterns, domain reputation (WHOIS), and obfuscation techniques.
- **Real-World Datasets**: Trained on over 200,000 samples from:
    - **PhiUSIIL** (UCI Machine Learning Repository)
    - **Mendeley Data** (vfszbj9b36)
    - **PhishTank** (Live feed)
- **Entropy Analysis**: Detects random generated subdomains and paths used by algorithmic phishing kits.

### Web Application
- **User Authentication**: Secure JWT-based auth with bcrypt password hashing.
- **Role-Based Access Control**:
    - **Admin Dashboard**: Analytics, user management, threat logs, and system health.
    - **User Dashboard**: Personal scan history, profile management, and threat alerts.
- **Real-time Scanning**: Instant URL analysis with detailed threat breakdown.
- **Interactive Visualizations**: Charts and graphs for threat statistics using Chart.js.

## 🛠️ Tech Stack

### Backend
- **Core**: Python 3.10+, Flask 3.0
- **Database**: SQLite with SQLAlchemy ORM
- **ML Libraries**: scikit-learn, NumPy, Pandas, Joblib
- **Data Source APIs**: integration with UCI ML Repo (`ucimlrepo`)
- **Security**: PyJWT, bcrypt

### Frontend
- **Structure**: HTML5, Semantic Web
- **Styling**: Tailwind CSS (Utility-first)
- **Logic**: Modern Vanilla JavaScript (ES6+)
- **Charts**: Chart.js

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ABDULRAZACK1029/Phishing-Detection-System.git
   cd Phishing-Detection-System
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On Windows, you may need Microsoft Visual C++ Build Tools for scikit-learn.*

3. **Initialize the Database:**
   The SQLite database (`instance/phishguard.db`) is automatically created on the first run.

4. **Train the ML Model (Optional but Recommended):**
   The repo comes with a pre-trained model, but you can retrain it with the latest data:
   ```bash
   python ml/train_with_real_data.py
   ```
   This script will download datasets, train the ensemble, and save the best model to `ml/phishing_model.pkl`.

5. **Run the Application:**
   ```bash
   python app.py
   ```
   Access the app at `http://localhost:5000`.

## 🧪 Machine Learning Workflow

The ML core is located in the `ml/` directory:

1.  **Data Loading (`ml/multi_dataset_loader.py`)**: Fetches and merges data from UCI, Mendeley, and PhishTank.
2.  **Feature Extraction (`ml/feature_extraction.py`)**: Extracts lexical and host-based features from URLs.
3.  **Training (`ml/train_with_real_data.py`)**:
    *   Splits data into training/testing sets.
    *   Trains Random Forest and Gradient Boosting classifiers.
    *   Evaluates using 5-Fold Cross-Validation.
    *   Saves the best performing ensemble.
4.  **Prediction (`ml/predict.py`)**: Loads the model and serves predictions to the Flask app.

## 📂 Project Structure

```
.
├── app.py                 # Main Flask application entry point
├── auth.py                # JWT authentication logic
├── models.py              # Database models (User, ScanResult)
├── requirements.txt       # Dependencies
├── .gitignore             # Git ignore rules (includes large ML models)
├── instance/              # SQLite database storage
├── ml/                    # Machine Learning Module
│   ├── datasets/          # Cached CSV datasets (gitignored)
│   ├── feature_extraction.py # URL feature engineering
│   ├── improved_trainer.py   # Advanced model training logic
│   ├── multi_dataset_loader.py # Dataset integration
│   ├── train_with_real_data.py # Main training script
│   ├── predict.py         # Prediction interface
│   └── model_metadata.json # Model performance stats
├── routes/                # API Blueprints
│   ├── auth.py
│   ├── scan.py
│   ├── admin.py
│   └── user.py
└── templates/             # HTML Templates
    ├── admin_dashboard.html
    ├── user_dashboard.html
    ├── threat_logs.html
    └── ...
```

## 🔒 Default Credentials

*   **Admin Email**: `admin@phishguard.com`
*   **Password**: `admin123`

## 📊 API Endpoints

| Method | Endpoint | Description | Auth |
|:---|:---|:---|:---|
| `POST` | `/api/auth/login` | Authenticate user & get token | No |
| `POST` | `/api/scan/url` | Scan a URL for phishing | User |
| `GET` | `/api/scan/history` | Get past scan results | User |
| `GET` | `/api/admin/dashboard/stats` | System-wide statistics | Admin |
| `GET` | `/api/admin/threats` | Detailed threat logs | Admin |

## 📜 License

This project is licensed under the ISC License.
