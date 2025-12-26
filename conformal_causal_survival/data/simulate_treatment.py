import numpy as np
from sklearn.linear_model import LogisticRegression

def generate_treatment_melanoma(data, seed=42):
    """Generate treatment for Melanoma based on tumor characteristics."""
    np.random.seed(seed)
    
    # Check if columns exist, handle renaming if needed
    cols = ['age_years', 'thickness_mm', 'ulcer_present']
    for c in cols:
        if c not in data.columns:
            raise ValueError(f"Column {c} not found in Melanoma data")
            
    X = data[cols].copy()
    X['age_std'] = (X['age_years'] - X['age_years'].mean()) / X['age_years'].std()
    X['thick_std'] = (X['thickness_mm'] - X['thickness_mm'].mean()) / X['thickness_mm'].std()
    
    # True propensity: younger + thicker + ulcerated → wide excision
    # We want younger patients to be treated more often (negative coeff on age)
    # Thicker tumors -> more treatment (positive coeff)
    # Ulceration -> more treatment (positive coeff)
    logit_ps = (-0.5 - 0.3 * X['age_std'] + 0.4 * X['thick_std'] + 
                0.5 * X['ulcer_present'])
    ps_true = 1 / (1 + np.exp(-logit_ps))
    
    treatment = np.random.binomial(1, ps_true)
    return treatment, ps_true

def generate_treatment_follicular(data, seed=42):
    """Generate treatment for Follicular based on stage and age."""
    np.random.seed(seed)
    
    cols = ['age_years', 'stage', 'hemoglobin']
    for c in cols:
         if c not in data.columns:
             raise ValueError(f"Column {c} not found in Follicular data")

    X = data[cols].copy()
    X['age_std'] = (X['age_years'] - X['age_years'].mean()) / X['age_years'].std()
    X['hgb_std'] = (X['hemoglobin'] - X['hemoglobin'].mean()) / X['hemoglobin'].std()
    
    # Advanced stage + lower hemoglobin → combined therapy
    # stage is often a factor or string, need to ensure it's numeric or comparable
    # In randomForestSRC follic data, stage is distinct. 
    # If it is numeric 1-4 or similar:
    # We assume 'stage' is already numeric or we convert it.
    
    is_advanced = (X['stage'] >= 3).astype(int)
    
    logit_ps = (-0.2 + 0.6 * is_advanced - 
                0.3 * X['hgb_std'] - 0.2 * X['age_std'])
    ps_true = 1 / (1 + np.exp(-logit_ps))
    
    treatment = np.random.binomial(1, ps_true)
    return treatment, ps_true
