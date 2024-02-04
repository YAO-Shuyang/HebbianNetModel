import numpy as np
from scipy.optimize import leastsq
from scipy.stats import poisson, ks_2samp
from tqdm import tqdm

def _poisson_res(l: float, x: np.ndarray, y: np.ndarray):
    return y - poisson.pmf(x, l)

def poisson_fit(x, y, l0: float = 5) -> float:
    """
    Poisson fit: Fit the x, y with the given lambda
    """
    para = leastsq(_poisson_res, x0 = l0, args = (x, y))
    return para[0][0]

def poisson_kstest(input: np.ndarray, monte_carlo_times: int=10000):
    max_num = int(np.max(input))
    x = np.arange(1, max_num+1)
    a = np.histogram(input, bins=max_num, range=(0.5, max_num+0.5))[0]
    prob = a / np.nansum(a)
    # Fit Poisson distribution
    lam = poisson_fit(x, prob)
    P0 = poisson.rvs(lam, size=len(input))
    D0 = ks_2samp(input, P0)[0]
    
    D = np.zeros(monte_carlo_times, dtype=np.float64)
    P = P0
    # It imposes caveats (Lilliefors et al., 1967) for KS test while the 
    # parameters of this distribution are estimated from the input itself.
    # We uses a monte-carlo test to solve this problem.
    for i in tqdm(range(monte_carlo_times)):
        x = np.arange(1, max_num+1)
        a = np.histogram(P, bins=max_num, range=(0.5, max_num+0.5))[0]
        prob = a / np.nansum(a)
        lam = poisson_fit(x, prob)
        Pt = poisson.rvs(lam, size=len(P))
        D[i] = ks_2samp(P, Pt)[0]
        P = Pt
    
    return D0, np.where(D0 < D)[0].shape[0] / monte_carlo_times