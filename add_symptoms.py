"""Add new symptoms to the comprehensive dataset."""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "lung_cancer_comprehensive.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "lung_cancer_comprehensive.csv"

# New symptoms to add
NEW_SYMPTOMS = [
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

def add_symptoms(df: pd.DataFrame, diagnosis_col: str) -> pd.DataFrame:
    """Add new symptom columns with realistic patterns based on diagnosis."""
    np.random.seed(42)
    
    for symptom in NEW_SYMPTOMS:
        # Create symptoms that are more likely in cancer cases
        # Cancer patients (diagnosis == 1) have 60-90% chance of symptom
        # Non-cancer (diagnosis == 0) have 5-15% chance
        symptom_values = []
        for diag in df[diagnosis_col]:
            if diag == 1:
                # Cancer cases: higher probability of symptoms
                prob = np.random.uniform(0.6, 0.9)
            else:
                # Non-cancer: lower probability
                prob = np.random.uniform(0.05, 0.15)
            symptom_values.append(1 if np.random.random() < prob else 0)
        
        df[symptom] = symptom_values
    
    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    df = add_symptoms(df, "diagnosis")
    
    # Reorder columns: demographics, lifestyle, medical history, symptoms, diagnosis
    existing_symptom_cols = [
        "coughing", "fatigue", "weight_loss", "shortness_of_breath", "chest_pain",
        "wheezing", "yellow_fingers", "clubbing"
    ]
    
    non_symptom_cols = [
        col for col in df.columns 
        if col not in existing_symptom_cols 
        and col not in NEW_SYMPTOMS 
        and col != "diagnosis"
    ]
    
    # Reorder all columns: non-symptoms, existing symptoms, new symptoms, diagnosis
    new_column_order = (
        non_symptom_cols + 
        existing_symptom_cols + 
        NEW_SYMPTOMS + 
        ["diagnosis"]
    )
    
    # Remove duplicates while preserving order
    seen = set()
    unique_order = []
    for col in new_column_order:
        if col not in seen:
            seen.add(col)
            unique_order.append(col)
    
    df = df[unique_order]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nNew shape: {df.shape}")
    print(f"New columns: {list(df.columns)}")
    print(f"\nSaved to: {OUTPUT_CSV}")

