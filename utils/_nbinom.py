import numpy as np
from scipy.optimize import leastsq
from scipy.stats import nbinom, ks_2samp
from tqdm import tqdm

def _nbinom_res(params, x, y):
    r, p = params
    return y - nbinom.pmf(x, r, p)

def nbinom_fit(x, y, r0: float=0.5, p0: float=0.2):
    para, _ = leastsq(_nbinom_res, [r0, p0], args = (x, y))
    return para[0], para[1]

def nbinom_kstest(input: np.ndarray, monte_carlo_times: int=10000):
    """Using KS test to identify the goodness of fit of NB distribution
    """
    max_num = int(np.max(input))
    x = np.arange(1, max_num+1)
    a = np.histogram(input, bins=max_num, range=(0.5, max_num+0.5))[0]
    prob = a / np.nansum(a)

    r, p = nbinom_fit(x, prob)
    P0 = nbinom.rvs(n=r, p=p, size=len(input))
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
        r, p = nbinom_fit(x, prob)
        Pt = nbinom.rvs(n=r, p=p, size=len(P))
        D[i] = ks_2samp(P, Pt)[0]
        P = Pt
    
    return D0, np.where(D0 < D)[0].shape[0] / monte_carlo_times