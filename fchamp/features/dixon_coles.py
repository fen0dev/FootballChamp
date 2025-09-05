import numpy as np
from scipy.stats import poisson

def dc_score_matrix(lh: float, la: float, rho: float, max_goals: int) -> np.ndarray:
    gh = np.arange(0, max_goals + 1)
    ga = np.arange(0, max_goals + 1)
    
    ph = poisson.pmf(gh, mu=lh)
    pa = poisson.pmf(ga, mu=la)

    M = np.outer(ph, pa)

    # Local correction for low points
    M[0, 0] *= (1 - lh * la * rho)

    if max_goals >= 1:
        M[0, 1] *= (1 + rho * (la - lh))
        M[1, 0] *= (1 + rho * (lh - la))
        M[1, 1] *= (1 - rho)
        
    M = np.maximum(M, 0)
    
    return M / M.sum()