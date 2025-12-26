import numpy as np
import pandas as pd

def construct_prediction_intervals(
    test_data, 
    cif_model_target, 
    conformal_quantile,
    times,
    cause
):
    """
    Construct conformalized prediction intervals for test set.
    
    For each test observation, find times t such that:
    V(X, A, t) <= q_{1-alpha}
    
    Where V(X, A, t) = 1 / CIF_j(t|X,A) (using the score definition).
    
    So we need 1 / CIF <= q
    => CIF >= 1/q
    
    Wait, if q is from calibration scores where high score = bad fit (unobserved event).
    If we want coverage, we include t where the score is "acceptably low".
    
    Score(t) = 1/CIF(t).
    Condition: 1/CIF(t) <= q  => CIF(t) >= 1/q.
    
    This implies we predict the event will happen in [t_start, t_end] where CIF is large enough?
    But CIF is monotonic increasing. So CIF(t) >= 1/q corresponds to t >= t*.
    So interval is [t*, max_time].
    
    This makes sense: "The event will occur after t*".
    (Since CIF(0)=0, score is infinite at 0).
    As t increases, CIF increases, Score decreases.
    Once Score <= q, we include t.
    """
    n_test = len(test_data)
    intervals = []
    
    threshold = 1.0 / (conformal_quantile + 1e-6)
    # So we include times where CIF(t) >= threshold.
    
    # If the quantile is very large (conservative), threshold is small.
    # We include almost all times.
    
    for i in range(n_test):
        obs = test_data.iloc[[i]]
        
        # Get predicted CIF curve
        cif_curve_df = cif_model_target.predict_cumulative_incidence(obs, times)
        cif_values = cif_curve_df.iloc[0].values
        
        # Times where CIF >= threshold
        valid_indices = np.where(cif_values >= threshold)[0]
        
        if len(valid_indices) > 0:
            lower_idx = valid_indices[0]
            upper_idx = valid_indices[-1]  # Typically the last time point
            
            lower = times[lower_idx]
            upper = times[upper_idx]
        else:
            # If threshold is never met (CIF too low everywhere)
            # We predict "never" or empty set?
            # Or conservative: entire range?
            # If we are unsure, maybe cover everything?
            # But the condition V <= q is never met, so empty set.
            # However, for survival, usually we want to cover T.
            # If CIF is low, T is likely large (or > max_time).
            # So maybe interval is (max_time, inf)?
            # Let's set to [max_time, max_time] or similar.
            lower = times[-1]
            upper = times[-1]
            
        intervals.append({
            'obs_id': obs.index[0],
            'lower': lower,
            'upper': upper,
            'width': upper - lower,
            'predicted_risk_final': cif_values[-1]
        })
    
    return pd.DataFrame(intervals)
