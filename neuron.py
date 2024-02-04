import torch
import numpy as np
from scipy.stats import gamma, poisson, expon
import copy as cp

def _gaussion(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

class PlaceCells:
    def __init__(self, n_neuron: int, input_vec_length: int, track_len: float = 600, sigma: float = 8):
        # define peak rate which follows a Gamma distribution with parameter estimated from real distribution.
        #   alpha = 1.695, loc = 0.418, scale = 0.547 # 10227
        peak_rates = gamma.rvs(1.695, loc=0.418, scale=0.547, size=n_neuron)
        
        # define field size which follows a log-normal distribution with parameter estimated from real distribution.
        #   mu = 49.42, sigma = 0.66 # 10209
        #sigmas = np.random.lognormal(0.4942, sigma=0.66, size=n_neuron)*2

        x_center = np.linspace(0, track_len, n_neuron)
        x_pos = np.linspace(0, track_len, input_vec_length)
        
        self.spatial_map = np.zeros((n_neuron, input_vec_length))
        for i in range(n_neuron):
            self.spatial_map[i, :] = _gaussion(x_pos, x_center[i], sigma+(np.random.rand()-0.5)*2)
            self.spatial_map[i, :] = self.spatial_map[i, :] / np.nanmax(self.spatial_map[i, :]) * peak_rates[i]
            
        self.t = x_pos
        self.peak_rates = peak_rates
        #self.sigmas = sigmas-
        
    def lapwise_simulate(self, tau: float = 2, n_lap: int = 40):
        """Based on paper Can et al., 2021, CA3 place fields emerge faster than 
        CA1 place fields when mice were re-exposed to the a familiarized environment.
        
        That is to say, not all the neurons display place fields at the first lap.
        The recovery of place fields, in terms of the cumulative fraction of fields
        that reappeared, was examined following exponential distribution 
        (Sheffield et al., 2017; Priestley et al., 2022; Can et al., 2021)
        
        Parameter
        ---------
        tau : float
            The exponential decay time constant.
        n_lap : int
            The number of laps to be simulated.
        """
        x = np.arange(1, n_lap+1, 1)
        y = (expon.cdf(x, scale=tau)*self.spatial_map.shape[0] // 1).astype(np.int64)
        order = np.arange(self.spatial_map.shape[0])
        np.random.shuffle(order)
        self.order = order
        self.border = y
        print(self.border)
    
    def get_map(self, lap: int):
        """
        Input current lap to obtain a current spatial map
        """
        I = cp.deepcopy(self.spatial_map)
        I[self.order[self.border[lap]:], :] = 0
        return I
        

class PlateuSignal:
    """Simulate the plateu signal
    """
    def __init__(self, n_neuron: int, input_vec_length: int, n_plateu: int, track_len: float = 600, sigma=50) -> None:
        x_center = (np.linspace(0, track_len, n_plateu)//1).astype(np.int64)
        x_pos = np.linspace(0, track_len, input_vec_length)
        
        self.plateu_signal = np.zeros((n_neuron, input_vec_length))
        ns = poisson.rvs(n_plateu/n_neuron, size=n_neuron)
        
        for i in range(n_neuron):
            for n in range(ns[i]):
                sigma_t = (sigma+(np.random.rand()-0.5)*20)
                _bot = int(np.random.rand()*track_len - sigma_t/2)
                _bot = _bot if _bot >= 0 else 0
                _top = int(_bot + sigma_t) if _bot + sigma_t <= input_vec_length else input_vec_length
                self.plateu_signal[i, _bot:_top] = 1
            
        self.t = x_pos


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    place_cells = PlaceCells(n_neuron=1000, input_vec_length=500, track_len=600)
    plateu_input = PlateuSignal(n_neuron=1000, input_vec_length=500, n_plateu=10000)
    
    i = 400
    plt.plot(plateu_input.t, plateu_input.plateu_signal[i, :])
    plt.show()
        