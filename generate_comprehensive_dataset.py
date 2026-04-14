"""Generate comprehensive dataset with all symptoms included."""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = PROJECT_ROOT / "data" / "lung_cancer_comprehensive.csv"

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
N_SAMPLES = 150

# All symptoms (existing + new)
ALL_SYMPTOMS = [
    "coughing",
    "fatigue",
    "weight_loss",
    "shortness_of_breath",
    "chest_pain",
    "wheezing",
    "yellow_fingers",
    "clubbing",
    "hemoptysis",  # Coughing up blood
    "hoarseness",  # Voice changes
    "loss_of_appetite",
    "night_sweats",
    "fever",
    "difficulty_swallowing",
    "bone_pain",
    "headache",
    "anemia",
]

def generate_sample(has_cancer: bool) -> dict:
    """Generate a single sample with realistic correlations."""
    if has_cancer:
        age = np.random.randint(55, 75)
        smoking_years = np.random.randint(20, 50)
        air_pollution = np.random.randint(70, 100)
        alcohol_intake = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        bmi = np.random.uniform(27, 33)
        family_history = np.random.choice([0, 1], p=[0.3, 0.7])
        asbestos_exposure = np.random.choice([0, 1], p=[0.2, 0.8])
        occupational_exposure = np.random.choice([0, 1], p=[0.2, 0.8])
        copd = np.random.choice([0, 1], p=[0.3, 0.7])
        previous_lung_disease = np.random.choice([0, 1], p=[0.4, 0.6])
        physical_activity = np.random.choice([1, 2], p=[0.7, 0.3])
        
        # Cancer patients have more symptoms (60-90% probability)
        symptoms = {}
        for symptom in ALL_SYMPTOMS:
            prob = np.random.uniform(0.6, 0.9)
            symptoms[symptom] = 1 if np.random.random() < prob else 0
    else:
        age = np.random.randint(35, 55)
        smoking_years = np.random.randint(0, 20)
        air_pollution = np.random.randint(20, 60)
        alcohol_intake = np.random.choice([1, 2, 4], p=[0.5, 0.3, 0.2])
        bmi = np.random.uniform(20, 27)
        family_history = np.random.choice([0, 1], p=[0.8, 0.2])
        asbestos_exposure = np.random.choice([0, 1], p=[0.95, 0.05])
        occupational_exposure = np.random.choice([0, 1], p=[0.9, 0.1])
        copd = np.random.choice([0, 1], p=[0.9, 0.1])
        previous_lung_disease = np.random.choice([0, 1], p=[0.95, 0.05])
        physical_activity = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
        
        # Non-cancer patients have fewer symptoms (5-15% probability)
        symptoms = {}
        for symptom in ALL_SYMPTOMS:
            prob = np.random.uniform(0.05, 0.15)
            symptoms[symptom] = 1 if np.random.random() < prob else 0
    
    sample = {
        "age": age,
        "gender": np.random.choice([0, 1]),
        "smoking_years": smoking_years,
        "air_pollution": air_pollution,
        "alcohol_intake": alcohol_intake,
        "bmi": round(bmi, 1),
        "family_history": family_history,
        "asbestos_exposure": asbestos_exposure,
        "occupational_exposure": occupational_exposure,
        "copd": copd,
        "previous_lung_disease": previous_lung_disease,
        "physical_activity": physical_activity,
        **symptoms,
        "diagnosis": 1 if has_cancer else 0,
    }
    
    return sample

def main():
    """Generate comprehensive dataset."""
    # Generate approximately balanced dataset
    n_cancer = N_SAMPLES // 2
    n_no_cancer = N_SAMPLES - n_cancer
    
    data = []
    data.extend([generate_sample(True) for _ in range(n_cancer)])
    data.extend([generate_sample(False) for _ in range(n_no_cancer)])
    
    # Shuffle
    np.random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns
    non_symptom_cols = [
        "age", "gender", "smoking_years", "air_pollution", "alcohol_intake",
        "bmi", "family_history", "asbestos_exposure", "occupational_exposure",
        "copd", "previous_lung_disease", "physical_activity"
    ]
    column_order = non_symptom_cols + ALL_SYMPTOMS + ["diagnosis"]
    df = df[column_order]
    
    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated dataset with {len(df)} samples")
    print(f"Features: {len(df.columns) - 1} (excluding diagnosis)")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"\nClass distribution:")
    print(df["diagnosis"].value_counts())

if __name__ == "__main__":
    main()

