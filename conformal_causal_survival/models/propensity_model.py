from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

class PropensityScoreModel:
    """
    Model treatment assignment π(a|x) = P(A=a | X=x).
    """
    def __init__(self, features, method='gbm'):
        self.features = features
        self.method = method
        self.model = None
        
    def fit(self, data):
        """Fit propensity score model."""
        X = data[self.features]
        A = data['treatment']
        
        if self.method == 'gbm':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42
            )
        else:
            self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
            
        self.model.fit(X, A)
        
    def predict_propensity(self, data):
        """
        Predict P(A=1|X) for each observation.
        """
        X = data[self.features]
        # Return probability of class 1
        ps = self.model.predict_proba(X)[:, 1]
        return ps
