import numpy as np

def evaluate_coverage(test_data, prediction_intervals, alpha=0.1):
    """
    Check if actual observed times fall within prediction intervals.
    
    Coverage = proportion of test observations where T_i ∈ [L_i, U_i]
    For censored data, we can only check if L_i <= C_i.
    If T_i is censored at C_i, and [L, U] covers [0, C_i] fully, we don't know.
    Standard approach: Evaluate only on observed events (conditional coverage)?
    Or use IPCW coverage.
    
    Simple approach (Guide):
    "n_valid += 1 if event == 1"
    """
    n_covered = 0
    n_valid = 0  # Only count uncensored observations of interest
    
    # prediction_intervals has 'obs_id' linking to test_data index
    # We Iterate
    
    # Ideally join on index
    combined = prediction_intervals.set_index('obs_id').join(test_data)
    
    # Filter for observed events of relevant type (assumed handled by caller)
    # The caller passed intervals for a specific cause.
    # We verify coverage for those who honestly experienced that event.
    
    # Note: If we just check coverage on observed events, it's biased.
    # But for "Implementation Guide", we follow the provided logic.
    
    # The logic in guide:
    # if obs['event'] == 1:
    #    if lower <= time <= upper: covered
    
    # We should assume 'event' column matches the cause we constructed intervals for?
    # Or strict 'event==1' means cause 1?
    # Let's assume input test_data has 'event' where 1 is the cause of interest.
    
    # If cause was passed as argument, we should filter by it.
    # But here we just assume the relevant events are '1'.
    
    relevant_events = combined[combined['event'] == 1]
    n_valid = len(relevant_events)
    
    if n_valid == 0:
        return {'coverage': 0.0, 'n_valid': 0}
        
    covered = (relevant_events['time_days'] >= relevant_events['lower']) & \
              (relevant_events['time_days'] <= relevant_events['upper'])
              
    n_covered = covered.sum()
    
    return {
        'coverage': n_covered / n_valid,
        'n_covered': n_covered,
        'n_valid': n_valid,
        'theoretical_coverage': 1 - alpha
    }
