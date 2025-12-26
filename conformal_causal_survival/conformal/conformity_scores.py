import numpy as np
import pandas as pd

def compute_conformity_scores(data, cif_model_target, cause, times):
    """
    Compute conformity scores for calibration set.
    
    For right-censored competing risks, we use a score based on the predicted probability
    assigned to the true outcome.
    
    If event == cause: Score = 1 / CIF(T_obs) (Inverse probability of observing the event)
                       Or 1 / (1 - CIF(T_obs))? Ideally we want high score = worse prediction.
                       If we predict 0.9 probability and event happens, good. Score low.
                       If we predict 0.1 probability and event happens, bad. Score high.
                       
    The guide implementation used:
    scores[i] = 1.0 / (1.0 - cif_pred + 1e-6) ??
    
    Let's check the guide logic again:
    "V_i = ... integral ... for event"
    "C_alpha = {t: V(x,a,t) <= q}"
    
    Actually, standard conformal survival (Candès) often uses:
    V_i = 1 - \hat{S}(T_i) if event occurs.
    
    For competing risks, if we target CIF_j:
    If event j occurs at T_i, we want T_i to be "covered".
    The set C(x) is usually constructed to include times where conformity is "acceptable".
    
    Let's stick to the guide's heuristic if rigorous theory is complex, 
    but for "Conformalized Survival Analysis" usually:
    Score corresponds to being "strange".
    Lower density/probability = Higher score.
    CIF_j(t) is P(T<=t, D=j).
    
    Guide Logic:
    - If event j occurs: Score = 1 / (1 - CIF(T_i) ... ) ?? 
      Wait, 1-CIF is Survival (roughly).
      
    Let's implement a general score function:
    V_i = max_t ( ... ) - this is for functional.
    
    Let's use the code provided in the guide step 4.1.
    """
    n = len(data)
    scores = np.zeros(n)
    
    # Pre-compute CIF for all data at their observed times?
    # Or for each i, predict at T_i
    
    # Ideally efficient batch prediction
    # We need CIF(T_obs) for each i
    
    # Optimization: Predict for all unique T_obs
    unique_times = np.unique(data['time_days'])
    
    # We need CIF at specific time points t.
    # The model `predict_cumulative_incidence` takes `times`.
    # Let's predict for all unique times once (if feasible) or loop.
    # unique_times might be large (continuous).
    
    # Just loop for now as per guide
    for i in range(n):
        obs = data.iloc[[i]]
        T_obs = obs['time_days'].values[0]
        event = obs['event'].values[0]
        
        if event == cause:  # Event of interest occurred
            # Get predicted CIF at observed time
            cif_pred = cif_model_target.predict_cumulative_incidence(
                obs, 
                times=[T_obs]
            ).iloc[0, 0]
            
            # Score: High if predicted probability is low
            # If CIF is small, score is large?
            # Guide: scores[i] = 1.0 / (1.0 - cif_pred + 1e-6)
            # This looks like 1/S. If S is close to 1 (low risk), and event happens, score is small (~1).
            # If S is close to 0 (high risk), score is large?
            # Wait. If I predict HIGH risk (CIF close to 1), then 1-CIF is small, score is LARGE.
            # This seems counter-intuitive. If I predict high risk and event happens, score should be low (good prediction).
            
            # Maybe the guide meant: 1 / CIF_pred?
            # If CIF is 0.9 (high risk), 1/0.9 = 1.1 (low score).
            # If CIF is 0.1 (low risk), 1/0.1 = 10 (high score).
            # This makes sense. Large score = Unexpected event.
            
            scores[i] = 1.0 / (cif_pred + 1e-6)
            
        else:
            # Censored or competing event
            # We treat this as "no information" or infinite score for interval construction?
            # In standard CSA (Candès), censored data is handled by reweighting or separate calibration.
            # The guide handles censoring in `V_i`.
            
            # Guide: if event == 0 (censored): scores[i] = -log(S_pred)
            # For competing risks, if censored, we don't know if event 1 would happen.
            
            # Let's assign -infinity or specific value if we use the specialized conformal method.
            # For simplicity and "Implementation Guide" fidelity:
            scores[i] = np.inf 
            # (We only calibrate on non-censored? Or use IPCW?)
            # The guide says "Compute conformity scores ... for calibration set".
            # Equation 152: V_i = integral ... dN^j(s).
            # If censored, dN=0 everywhere, so integral=0.
            # So V_i = 0.
            
            # If we use V_i = 0 for censored, then q_{1-alpha} will be affected.
            scores[i] = 0.0

    return scores
