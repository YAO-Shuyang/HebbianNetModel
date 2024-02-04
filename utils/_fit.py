import numpy as np
from scipy.stats import gamma, lognorm
from model.utils._poisson import poisson_fit
from model.utils._nbinom import nbinom_fit


def fit_field_size(field_size: np.ndarray, fit_model: str = 'lognorm') -> tuple[float, float, float]:
    """
    Fit the simulated field sizes with the given model.
    
    According to a study conducted on fruit bats (Eliav et al., 2021), multiple field
    sizes is a critical feature for hippocampal spatial coding in realistic environments.
    The field size was reported as a log-normal distribution, although gamma distribution
    was also used. This function fits the simulated field size with the given statistic
    distribution model.
    
    Parameters
    ----------
    field_size : np.ndarray
        A numpy array that contains the simulated field size for each neuron.
    fit_model : str, optional
        The candidate statistic distribution model: 'gamma', 'lognorm'.
        The default is 'lognorm'.
        
    return
    ------
    tuple: parameter tuple that contains 3 parameters: (shape, loc, scale)
    """
    if fit_model == 'gamma':
        return gamma.fit(field_size)
    elif fit_model == 'lognorm':
        return lognorm.fit(field_size)
    else:
        raise NotImplementedError(
            f"fit_model should be 'gamma' or 'lognorm', not {fit_model}"
        )
        


def fit_field_number(field_number: np.ndarray, fit_model: str = 'poisson', **kwargs) -> float or tuple[float, float, float]:
    """
    Fit the simulated field numbers with the given model.
    
    Prior findings (Rich et al., 2014, 2020) suggest that the distribution of
    the number of fields is negative binomial. However, we observed a poisson-
    featured distribution (could be co-fitted by nbinom and poisson distr.)
    
    Parameters
    ----------
    field_number : np.ndarray
        A numpy array that contains the simulated field number for each neuron.
    fit_model : str, optional
        The candidate statistic distribution model: 'poisson', 'nbinom'.
        The default is 'poisson'.
        
    return
    ------
    tuple: parameter tuple that contains 1 or 2 parameters
        lambda, or the rate parameter for Poissson distribution if fit_model 
        is 'poisson'.
        r and p, the shape parameter and scale parameter for negative binomial 
        distribution if fit_model is 'nbinom'.
    """
    max_num = int(np.max(field_number))
    x = np.arange(1, max_num+1)
    y = np.histogram(field_number, bins=max_num, range=(0.5, max_num+0.5))[0]
    if fit_model == 'poisson':
        return poisson_fit(x, y, **kwargs)
    elif fit_model == 'nbinom':
        return nbinom_fit(x, y, **kwargs)
    else:
        raise NotImplementedError(
            f"fit_model should be 'poisson' or 'nbinom', not {fit_model}"
        )