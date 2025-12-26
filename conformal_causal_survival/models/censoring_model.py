from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
import pandas as pd

class CensoringModel:
    """
    Model censoring mechanism G(t|x,a) = P(C > t | X=x, A=a).
    """
    def __init__(self, features):
        self.features = features
        self.model = None
        
    def fit(self, data):
        """
        Fit censoring model using reverse-time approach.
        Treat censoring as "event" and actual events as "censored".
        """
        # Create censoring indicator
        # Event = 1 if original event was 0 (censored), else 0
        y_censored = (data['event'] == 0).astype(bool)
        y_time = data['time_days']
        
        X = data[self.features]
        
        y = np.array(
            list(zip(y_censored, y_time)),
            dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
        )
        
        self.model = CoxPHSurvivalAnalysis(alpha=1e-4) # Fixed small penalty for stability
        self.model.fit(X, y)
        
    def predict_censoring_survival(self, data, times):
        """
        Predict P(C > t | X, A) for given times.
        """
        X = data[self.features]
        surv_funcs = self.model.predict_survival_function(X)
        
        results = []
        for fn in surv_funcs:
            # Evaluate at requested times
            vals = fn(times)
            results.append(vals)
            
        # Return as DataFrame where columns are times
        # Handle if vals is smaller than times (if time > max observed)
        # linear extension or constant
        return pd.DataFrame(results, columns=times)
