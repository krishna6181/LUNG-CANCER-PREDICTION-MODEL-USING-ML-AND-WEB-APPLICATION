#!/usr/bin/env python3
"""
Test script to verify Flask application startup and basic functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

def test_flask_startup():
    """Test if Flask app can start without errors."""
    print("Testing Flask application startup...")

    try:
        from app.main import app

        # Test app configuration
        assert app.config['SECRET_KEY'] is not None, "Secret key not set"
        assert 'lung_cancer_pipeline.joblib' in str(app.config.get('MODEL_PATH', '')), "Model path not configured"

        print("✓ Flask app configuration is valid")

        # Test basic routes
        with app.test_client() as client:
            # Test home page redirect
            response = client.get('/')
            assert response.status_code in [200, 302], f"Home page failed: {response.status_code}"
            print("✓ Home page accessible")

            # Test sign-in page
            response = client.get('/signin')
            assert response.status_code == 200, f"Sign-in page failed: {response.status_code}"
            print("✓ Sign-in page accessible")

            # Test sign-up page
            response = client.get('/signup')
            assert response.status_code == 200, f"Sign-up page failed: {response.status_code}"
            print("✓ Sign-up page accessible")

        return True

    except Exception as e:
        print(f"✗ Flask startup test failed: {e}")
        return False

def test_database():
    """Test database connectivity."""
    print("\nTesting database connectivity...")

    try:
        import sqlite3
        from app.auth import init_db

        # Initialize database
        init_db()

        # Test database operations
        from app.auth import create_user, verify_user

        # Create test user
        success = create_user("testuser", "test@example.com", "password123")
        if success:
            print("✓ User creation successful")

            # Verify user
            user = verify_user("testuser", "password123")
            if user:
                print("✓ User verification successful")
                return True
            else:
                print("✗ User verification failed")
                return False
        else:
            print("✗ User creation failed")
            return False

    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_model_integration():
    """Test model integration with Flask app."""
    print("\nTesting model integration...")

    try:
        from app.main import load_pipeline

        pipeline = load_pipeline()
        print("✓ Model integration successful")

        # Test prediction with sample data
        import pandas as pd
        from app.main import prepare_features

        sample_data = {
            'age': '65', 'gender': '1', 'smoking_years': '40',
            'air_pollution': '85', 'alcohol_intake': '3', 'bmi': '28.5',
            'family_history': '1', 'asbestos_exposure': '1',
            'occupational_exposure': '1', 'copd': '1',
            'previous_lung_disease': '0', 'physical_activity': '2',
            'coughing': '1', 'fatigue': '1', 'weight_loss': '1',
            'shortness_of_breath': '1', 'chest_pain': '1', 'wheezing': '1',
            'yellow_fingers': '1', 'clubbing': '1', 'hemoptysis': '0',
            'hoarseness': '0', 'loss_of_appetite': '1', 'night_sweats': '0',
            'fever': '0', 'difficulty_swallowing': '0', 'bone_pain': '0',
            'headache': '0', 'anemia': '1'
        }

        df = prepare_features(sample_data)
        prediction = pipeline.predict(df)
        probability = pipeline.predict_proba(df)

        print(f"✓ Sample prediction: {prediction[0]}")
        print(f"✓ Sample probability: {probability[0][1]:.3f}")
        return True

    except Exception as e:
        print(f"✗ Model integration test failed: {e}")
        return False

def main():
    """Run all Flask application tests."""
    print("🧪 Flask Application Test Suite")
    print("=" * 50)

    tests = [
        test_flask_startup,
        test_database,
        test_model_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Flask Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Flask tests passed! The web application is ready to run.")
        print("\nTo start the application:")
        print("  cd /path/to/project")
        print("  flask --app app.main run --debug")
        print("  Open http://127.0.0.1:5000 in your browser")
        return 0
    else:
        print("❌ Some Flask tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
