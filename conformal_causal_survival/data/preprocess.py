from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def create_splits(data, test_size=0.5, random_state=42):
    """
    Split data into train/calibration/test (50%/25%/25%).
    Stratified by event type to maintain event distribution.
    
    Args:
        data: DataFrame containing 'event' column
        test_size: Proportion of data for test+calibration (default 0.5)
        random_state: Seed for reproducibility
    
    Returns:
        train_data, calib_data, test_data
    """
    # Stratify by event to ensure all event types are represented
    # If event counts are too small, fallback to no stratification
    stratify = data['event']
    if stratify.value_counts().min() < 2:
        stratify = None
        
    train_data, temp_data = train_test_split(
        data, 
        test_size=test_size, 
        stratify=stratify,
        random_state=random_state
    )
    
    # Stratify split for calibration/test
    stratify_temp = temp_data['event']
    if stratify_temp.value_counts().min() < 2:
        stratify_temp = None

    calib_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        stratify=stratify_temp,
        random_state=random_state
    )
    
    return train_data, calib_data, test_data

def check_positivity(data, treatment_col='treatment', min_prob=0.01):
    """
    Check positivity assumption: 0 < P(A=1|X) < 1.
    This is just a heuristic check based on counts.
    """
    mean_treatment = data[treatment_col].mean()
    if mean_treatment < min_prob or mean_treatment > (1 - min_prob):
        return False
    return True
