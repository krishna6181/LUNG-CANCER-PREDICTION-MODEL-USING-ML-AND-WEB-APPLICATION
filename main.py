from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from app.auth import create_user, get_user_by_id, verify_user


FEATURE_ORDER = [
    "age",
    "gender",
    "smoking_years",
    "air_pollution",
    "alcohol_intake",
    "bmi",
    "family_history",
    "asbestos_exposure",
    "occupational_exposure",
    "copd",
    "previous_lung_disease",
    "physical_activity",
    "coughing",
    "fatigue",
    "weight_loss",
    "shortness_of_breath",
    "chest_pain",
    "wheezing",
    "yellow_fingers",
    "clubbing",
    "hemoptysis",
    "hoarseness",
    "loss_of_appetite",
    "night_sweats",
    "fever",
    "difficulty_swallowing",
    "bone_pain",
    "headache",
    "anemia",
]

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "lung_cancer_pipeline.joblib"


def load_pipeline():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {ARTIFACT_PATH}. Run src/train_model.py first."
        )
    return joblib.load(ARTIFACT_PATH)


def prepare_features(source: dict[str, Any]) -> pd.DataFrame:
    """Return a single-row DataFrame so the pipeline keeps column names."""
    row: dict[str, list[float]] = {}
    for feature in FEATURE_ORDER:
        try:
            row[feature] = [float(source[feature])]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid value for {feature}") from exc
    return pd.DataFrame(row)


def classify_risk(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


app = Flask(__name__)
app.secret_key = "your-secret-key-change-in-production-12345"  # Change this in production!
pipeline = load_pipeline()


def login_required(f):
    """Decorator to require login for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("signin"))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/")
def home():
    """Redirect to dashboard if logged in, otherwise to signin."""
    if "user_id" in session:
        return redirect(url_for("index"))
    return redirect(url_for("signin"))


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def index():
    user = get_user_by_id(session["user_id"])
    context: dict[str, Any] = {
        "prediction": None,
        "probability": None,
        "risk_level": None,
        "user": user,
    }
    if request.method == "POST":
        try:
            feature_frame = prepare_features(request.form)
            proba = float(pipeline.predict_proba(feature_frame)[0][1])
            prediction = int(proba >= 0.5)
            context.update(
                {
                    "prediction": prediction,
                    "probability": round(proba, 3),
                    "risk_level": classify_risk(proba),
                }
            )
        except ValueError as err:
            context["error"] = str(err)
    return render_template("index.html", **context)


@app.post("/api/predict")
@login_required
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        feature_frame = prepare_features(payload)
        proba = float(pipeline.predict_proba(feature_frame)[0][1])
        return jsonify(
            {
                "probability": proba,
                "risk_level": classify_risk(proba),
                "prediction": int(proba >= 0.5),
            }
        )
    except ValueError as err:
        return jsonify({"error": str(err)}), 400


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """User registration page."""
    if "user_id" in session:
        return redirect(url_for("index"))
    
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        if not username or not email or not password:
            error = "All fields are required."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif len(password) < 6:
            error = "Password must be at least 6 characters long."
        else:
            if create_user(username, email, password):
                session["user_id"] = verify_user(username, password)["id"]
                return redirect(url_for("index"))
            else:
                error = "Username or email already exists."
    
    return render_template("signup.html", error=error)


@app.route("/signin", methods=["GET", "POST"])
def signin():
    """User login page."""
    if "user_id" in session:
        return redirect(url_for("index"))
    
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if not username or not password:
            error = "Username and password are required."
        else:
            user = verify_user(username, password)
            if user:
                session["user_id"] = user["id"]
                return redirect(url_for("index"))
            else:
                error = "Invalid username or password."
    
    return render_template("signin.html", error=error)


@app.route("/logout")
def logout():
    """Logout user and clear session."""
    session.clear()
    return redirect(url_for("signin"))


if __name__ == "__main__":
    app.run(debug=True)

