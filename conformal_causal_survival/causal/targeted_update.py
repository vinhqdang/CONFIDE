import numpy as np
from scipy.optimize import minimize_scalar

def tmle_update(initial_estimate, clever_covariate, observed_outcome):
    """
    Update initial estimate by fitting:
    logit(Q*(t)) = logit(Q(t)) + ε * H(t)
    
    where ε is chosen to solve the efficient influence function equation.
    
    Args:
        initial_estimate: Q(t), shape (n,)
        clever_covariate: H(t), shape (n,)
        observed_outcome: Y(t) or I(T<=t, delta=j), shape (n,)
    
    Returns:
        Q_targeted: Updated estimate
        epsilon: The fluctuation parameter
    """
    # Safe logit
    # Clip to avoid inf
    # Q must be in (0, 1)
    # initial_estimate expected to be CIF values
    
    Q = np.clip(initial_estimate, 1e-5, 1 - 1e-5)
    logit_Q = np.log(Q / (1 - Q))
    
    H = clever_covariate
    Y = observed_outcome
    
    def loss_function(epsilon):
        """
        Loss function based on logistic regression likelihood (approximated).
        Standard TMLE solves efficient influence function = 0.
        For logistic fluctuation, this corresponds to maximizing likelihood of 
        logistic regression with offset logit_Q and predictor H.
        """
        # Q_eps = expit(logit_Q + eps * H)
        logit_Q_eps = logit_Q + epsilon * H
        # Stable sigmoid
        # Q_eps = 1 / (1 + exp(-logit_Q_eps))
        
        # Binary cross entropy loss (negative log likelihood)
        # Using log-sum-exp trick implicitly or just simple formula
        # loss = - (Y * log(Q_eps) + (1-Y) * log(1-Q_eps))
        
        # More numerically stable:
        # log(Q_eps) = -log(1 + exp(-z)) = z - log(1 + exp(z))
        # log(1-Q_eps) = -z - log(1 + exp(-z)) ??
        
        # Let's use scipy.special.expit if available or implementation
        # Actually usually we solve sum H(Y - Q_eps) = 0.
        # But minimize_scalar requires a scalar loss.
        # The integral of the EIF or the likelihood function.
        
        # Simple logistic loss:
        # loss = sum(log(1 + exp(logit_Q_eps)) - Y * logit_Q_eps)
        
        val = np.logaddexp(0, logit_Q_eps) - Y * logit_Q_eps
        return np.sum(val)
    
    # Minimize loss
    result = minimize_scalar(loss_function, bounds=(-100, 100), method='bounded')
    epsilon_star = result.x
    
    # Compute targeted estimate
    logit_targeted = logit_Q + epsilon_star * H
    Q_targeted = 1 / (1 + np.exp(-logit_targeted))
    
    return Q_targeted, epsilon_star
