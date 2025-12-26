from sksurv.metrics import concordance_index_censored
import numpy as np

def evaluate_discrimination(test_data, cif_model, cause=1, time_horizon=None):
    """
    Evaluate discrimination using C-index.
    """
    # Create event indicator for cause of interest
    # Censored: event=0 or event=other
    # "Event" means event=cause
    
    # Correct way for C-index with competing risks is specific:
    # Usually we treat other events as censored for cause-specific C-index.
    
    event_indicator = (test_data['event'] == cause).astype(bool)
    event_times = test_data['time_days']
    
    # Create structured array for sksurv
    # status=True if event, False if censored
    
    # We need risk scores. Higher risk = lower survival time predicted.
    # CIF at horizon: Higher CIF = Higher risk.
    
    if time_horizon is None:
        time_horizon = test_data['time_days'].median()
        
    cif_pred = cif_model.predict_cumulative_incidence(
        test_data, 
        times=[time_horizon]
    ).iloc[:, 0].values
    
    # concordance_index_censored(event_indicator, event_time, estimate)
    # estimate: predicted risk. Higher value = shorter survival (higher risk).
    # So CIF is correct (higher CIF = higher risk).
    
    c_index = concordance_index_censored(
        event_indicator,
        event_times,
        cif_pred
    )
    
    return c_index[0]
