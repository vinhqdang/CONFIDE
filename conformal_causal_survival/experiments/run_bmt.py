import os
import sys
import numpy as np
import pandas as pd
import warnings

# Set R_HOME explicitly
if 'R_HOME' not in os.environ:
    os.environ['R_HOME'] = '/opt/miniconda3/envs/py313/lib/R'

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from conformal_causal_survival.data.load_datasets import DatasetLoader
from conformal_causal_survival.data.preprocess import create_splits
from conformal_causal_survival.models.nuisance_models import CauseSpecificHazardModel
from conformal_causal_survival.models.censoring_model import CensoringModel
from conformal_causal_survival.models.propensity_model import PropensityScoreModel
from conformal_causal_survival.conformal.conformity_scores import compute_conformity_scores
from conformal_causal_survival.conformal.calibration import get_conformal_quantile
from conformal_causal_survival.conformal.prediction_intervals import construct_prediction_intervals
from conformal_causal_survival.evaluation.coverage import evaluate_coverage
from conformal_causal_survival.evaluation.discrimination import evaluate_discrimination

def run_bmt_experiment():
    print("="*40, flush=True)
    print("RUNNING BMT EXPERIMENT (n=137)", flush=True)
    print("="*40, flush=True)
    
    print("Loading BMT Dataset...", flush=True)
    loader = DatasetLoader()
    try:
        data = loader.load_bmt()
    except Exception as e:
        print(f"Failed to load dataset: {e}", flush=True)
        return

    # Treatment should be present
    if 'treatment' not in data.columns:
        # Fallback simulation
        data['treatment'] = np.random.randint(0, 2, len(data))
    
    print("Creating Splits...", flush=True)
    train_data, calib_data, test_data = create_splits(data)

    # Ensure 'male' exists (some sources might use 'sex')
    if 'male' not in data.columns and 'sex' in data.columns:
         data['male'] = data['sex']
         
    feature_cols = ['age_years', 'male', 'treatment']
    # Check extra cols
    if 'disease' in data.columns:
        feature_cols.append('disease')
    
    # Simple encoding for disease if categorical
    # (Assuming loader handled it or SKSurv handles strings if categorical dtypes set)
    # Better to drop if complex strings for now to avoid errors
    # or ensure one-hot encoding.
    # SKSurv CoxPH supports pandas categorical/string? Not always.
    # Let's keep it simple: age, male, treatment
    safe_features = ['age_years', 'male', 'treatment']
    
    print("Training Cause-Specific Hazard Models...", flush=True)
    try:
        # Cause 1 = Relapse, Cause 2 = Death
        TARGET_CAUSE = 1 # Relapse
        
        model_c1 = CauseSpecificHazardModel(cause=1, features=safe_features, model_type='cox')
        model_c1.fit(train_data)
        
        model_c2 = CauseSpecificHazardModel(cause=2, features=safe_features, model_type='cox')
        model_c2.fit(train_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Training Failed: {e}", flush=True)
        return

    print("Training Censoring Model...", flush=True)
    cens_model = CensoringModel(features=safe_features)
    cens_model.fit(train_data)

    print("Training Propensity Score Model...", flush=True)
    ps_model = PropensityScoreModel(features=[c for c in safe_features if c != 'treatment'])
    ps_model.fit(train_data)
    
    times = np.quantile(data[data['event']==TARGET_CAUSE]['time_days'], np.linspace(0.1, 0.9, 10))
    times = np.sort(times)
    
    print("Computing Conformity Scores (Calibration)...", flush=True)
    calib_scores = compute_conformity_scores(calib_data, model_c1, cause=TARGET_CAUSE, times=times)
    
    alpha = 0.1
    print(f"Calibrating for alpha={alpha}...", flush=True)
    q_alpha = get_conformal_quantile(calib_scores, alpha)
    print(f"Conformal Quantile: {q_alpha}", flush=True)

    print("Constructing Prediction Intervals (Test)...", flush=True)
    intervals = construct_prediction_intervals(
        test_data, 
        model_c1, 
        q_alpha, 
        times, 
        cause=TARGET_CAUSE
    )
    
    print("Evaluating...", flush=True)
    cov_results = evaluate_coverage(test_data, intervals, alpha)
    c_index = evaluate_discrimination(test_data, model_c1, cause=TARGET_CAUSE)
    
    print("-" * 30, flush=True)
    print("RESULTS: BMT", flush=True)
    print("-" * 30, flush=True)
    print(f"Coverage: {cov_results['coverage']:.4f} (Valid Obs: {cov_results['n_valid']})", flush=True)
    print(f"C-index:  {c_index:.4f}", flush=True)
    print(f"Mean Width: {intervals['width'].mean():.2f} days", flush=True)
    print("-" * 30, flush=True)

    return {
        'dataset': 'BMT',
        'n': len(data),
        'coverage': cov_results['coverage'],
        'c_index': c_index,
        'width': intervals['width'].mean()
    }

if __name__ == "__main__":
    run_bmt_experiment()
