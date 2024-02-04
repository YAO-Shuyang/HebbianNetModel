from scipy.stats import gamma, ks_2samp
import numpy as np
from tqdm import tqdm

def gamma_fit(x: np.ndarray, **kwargs):
    return gamma.fit(x, **kwargs)

def gamma_kstest(input: np.ndarray, monte_carlo_times: int=10000):
    alpha, c, beta = gamma.fit(input)
    P0 = gamma.rvs(alpha, loc = c, scale=beta, size=len(input))
    D0 = ks_2samp(input, P0)[0]
    
    D = np.zeros(monte_carlo_times, dtype=np.float64)
    P = P0
    # It imposes caveats (Lilliefors et al., 1967) for KS test while the 
    # parameters of this distribution are estimated from the input itself.
    # We uses a monte-carlo test to solve this problem.
    for i in tqdm(range(monte_carlo_times)):
        #alpha, c, beta = gamma.fit(P, floc=0)
        Pt = gamma.rvs(alpha, loc=c, scale=beta, size=len(P))
        D[i] = ks_2samp(P, Pt)[0]
        P = Pt
        
    return D0, np.where(D0 < D)[0].shape[0] / monte_carlo_times