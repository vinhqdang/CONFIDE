from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
import numpy as np

class CauseSpecificHazardModel:
    """
    Fit separate models for each competing event.
    """
    def __init__(self, cause, features, model_type='cox'):
        self.cause = cause
        self.features = features
        self.model_type = model_type
        self.model = None
        
    def fit(self, data):
        """Fit model on training data."""
        # Create cause-specific event indicator
        # For cause j: event if status==j, censored otherwise
        # scikit-survival expects boolean event indicator
        
        X = data[self.features].copy()
        y_event = (data['event'] == self.cause).astype(bool)
        y_time = data['time_days']
        
        # Create structured array for sksurv
        y = np.array(
            list(zip(y_event, y_time)),
            dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
        )
        
        if self.model_type == 'cox':
            # Add small ridge penalty (alpha) for stability
            self.model = CoxPHSurvivalAnalysis(alpha=1e-4) # Fixed small penalty
        elif self.model_type == 'rsf':
            self.model = RandomSurvivalForest(n_estimators=100, random_state=42)
            
        if sum(y_event) > 0:
            cph = CoxPHFitter() if self.model_type == 'lifelines' else self.model
            if self.model_type == 'cox':
                 self.model.fit(X, y)
            elif self.model_type == 'rsf':
                 self.model.fit(X, y)
        else:
            print(f"Warning: No events for cause {self.cause}. Model not fit.")
            self.model = None
        
    def predict_cumulative_incidence(self, data, times):
        """
        Predict cumulative hazard/incidence.
        Note: scikit-survival predict_cumulative_hazard returns hazard function.
        We interpret this as approximation for CIF in simple Cause-Specific setting:
        CIF_j(t) = \int S(u-) d\Lambda_j(u)
        But for now, we just return the cumulative hazard or survival specific to this cause,
        which we will combine later.
        
        Actually, for TMLE we need hazard or survival probability.
        Let's return the survival function S_j(t) = exp(-H_j(t)) where H_j is cause-specific cumulative hazard.
        """

        if self.model is None:
             # Return zeros (no risk predicted)
             return pd.DataFrame(0.0, index=data.index, columns=times)
             
        X = data[self.features]
        
        # predict_survival_function returns array of functions or step functions
        surv_funcs = self.model.predict_survival_function(X)
        
        results = []
        for fn in surv_funcs:
            # Evaluate step function at specified times
            # fn(t) gives Survival probability S(t)
            # We approximate CIF ~ 1 - S(t) (Net failure probability)
            vals = 1.0 - fn(times)
            results.append(vals)
            
        return pd.DataFrame(results, columns=times)

    def predict_hazard(self, data):
        """Predict hazard risk score (linear predictor)."""
        X = data[self.features]
        return self.model.predict(X)
