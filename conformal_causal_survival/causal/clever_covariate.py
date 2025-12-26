import numpy as np

def compute_clever_covariate(data, ps, censoring_survival, treatment_value_to_target, times):
    """
    Compute clever covariate H(t|A,X) for a specific treatment arm.
    
    H(t) = [I(A=a) / P(A=a|X)] * [1 / P(C > t | A,X)]
    """
    n = len(data)
    n_times = len(times)
    
    # Treatment indicator
    A = data['treatment'].values
    I_A = (A == treatment_value_to_target).astype(float)
    
    # Propensity score for the target arm
    # ps input is P(A=1|X)
    if treatment_value_to_target == 1:
        g_A = ps
    else:
        g_A = 1.0 - ps
        
    # Clip propensity to avoid division by zero
    g_A = np.clip(g_A, 0.01, 0.99)
    
    # Censoring survival G(t|A,X)
    # censoring_survival is a DataFrame with columns=times, rows=observations
    # Ensure it's aligned
    G_t = censoring_survival.values
    # Clip censoring survival
    G_t = np.clip(G_t, 0.01, 1.0)
    
    # Compute inverse probability weight part
    # Shape: (n,)
    ipw = I_A / g_A
    
    # Broadcast to (n, n_times)
    # H = ipw / G_t
    H = ipw[:, np.newaxis] / G_t
    
    return H
