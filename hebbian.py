import torch
import numpy as np
from model.neuron import PlaceCells, PlateuSignal

N_CA3 = 2000
N_CA1 = 2000
N_PLATEU = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
torch.set_default_device(device)

def update_state(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, Y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """update_state: Update the status of neurons

    Parameters
    ----------
    X : torch.Tensor
        Input from CA3, typically from CA3 place cells with single fields but varying in sizes.
    W : torch.Tensor
        The weighted matrix of connected neuron.
    B : torch.Tensor
        Input that produce platue for synaptic connections.
    Y : torch.Tensor
        Output from CA1
    M : torch.Tensor
        The weighted matrix of connected lateral inhibitory connections.
    """
    Y = torch.matmul(torch.where(W > B, W, B), X)
    Y -= torch.matmul(M, Y)
    Y = torch.relu(Y)
    for i in range(Y.shape[0]):
        Y[i, :] = Y[i, :]/torch.max(Y[i, :])
    return Y

def hebbian_learning(W: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, eta: float = 0.01):
    # Implement the Hebbian learning rule
    # Δw = learning_rate (eta) * (post_activity (Y^T) outproduct pre_activity (X))
    
    outer = torch.zeros((W.shape[0], W.shape[1]))
    for i in range(X.shape[1]):
        outer += torch.outer(Y[:, i], X[:, i])
        
    outer /= X.shape[1]

    W += eta * (outer- W)
    return torch.where(W>=0, W, 0)

def anti_hebbian_learning(M: torch.Tensor, Y: torch.Tensor, eta: float = 0.001):
    # Implement the Anti-Hebbian learning rule
    # ΔM = - learning_rate (eta) * (outproduct of post_activity (y*y^T))
    outer = torch.zeros((M.shape[0], M.shape[1]))
    for i in range(Y.shape[1]):
        outer += torch.outer(Y[:, i], Y[:, i])
        
    outer /= Y.shape[1]
    
    M += eta * (outer - M)
    return torch.where(M>=0, M, 0)

def btsp_learning(B: torch.Tensor, X: torch.Tensor, I: torch.Tensor, eta: float = 0.01):
    # Implement the BTSP learning rule
    # ΔB = learning_rate (eta) * (post_plateu (I^T) * pre_activity (X))
    
    outer = torch.zeros((B.shape[0], B.shape[1]))
    for i in range(X.shape[1]):
        outer += torch.outer(I[:, i], X[:, i])
        
    outer /= X.shape[1]

    B += eta * (outer- B)
    return torch.where(B>=0, B, 0)

# Simulation parameters
N_LAP = 40  # Define the total number of laps
N_FRAME = 500 # Define the total frame number
N_CA3_TO_CA1 = 20 # Define the current active synapse number from CA3 to CA1
eta_e = 0.001 # learning rate of hebbian learning
eta_i = 0.0002 # learning rate of anti-hebbian learning
eta_b = 0.005 # learning rate of btsp learning

# Initial neuron states
PCs = PlaceCells(N_CA3, N_FRAME)
t = PCs.t
X = torch.from_numpy(PCs.spatial_map).float().to(device)
PlateuInputs = PlateuSignal(N_CA1, N_FRAME, N_PLATEU)
I = torch.from_numpy(PlateuInputs.plateu_signal).float().to(device)
Y = torch.zeros((N_CA1, N_FRAME))

M = torch.zeros((N_CA1, N_CA1))
W = torch.zeros((N_CA1, N_CA3))
B = torch.zeros((N_CA1, N_CA3))

for i in range(N_CA3):
    W[torch.randint(0, N_CA1, size=(N_CA3_TO_CA1,1)), i] = torch.rand(size=(N_CA3_TO_CA1,1))

import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("Spectral", N_LAP)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 8))
for l in range(N_LAP):
    print(l+1)
    # Update neuron states
    Y = update_state(X, W, B, Y, M)
    
    # Apply learning rules
    W = hebbian_learning(W, X, Y, eta=eta_e)
    M = anti_hebbian_learning(M, Y, eta=eta_i)
    B = btsp_learning(B, X, I, eta=eta_b)
    
    # Optionally: Record states and weights for analysis and visualization
    # ...
    
    y = Y.cpu().numpy()
    
    for i in range(5):
        for j in range(5):
            axes[i, j].plot(t, y[i*5+j, :], linewidth = 0.5, color=colors[l], alpha=0.5)

plt.show()

# Visualization and analysis
# ...
