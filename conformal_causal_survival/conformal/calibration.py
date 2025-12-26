import numpy as np

def get_conformal_quantile(scores, alpha=0.1):
    """
    Compute (1-α) quantile of calibration scores.
    
    For coverage level 1-α, we need the ⌈(n+1)(1-α)⌉/n quantile.
    Standard conformal prediction formula.
    """
    n = len(scores)
    val = (n + 1) * (1 - alpha)
    q_level = np.ceil(val) / n
    # Clip to [0, 1]
    q_level = np.clip(q_level, 0, 1)
    
    q_alpha = np.quantile(scores, q_level, method='higher') # Conservative 'higher'
    
    return q_alpha
