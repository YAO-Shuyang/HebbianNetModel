import torch
import numpy as np
from model.neuron import PlaceCells

N_CA3 = 2000
N_CA1 = 2000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
torch.set_default_device(device)

def update_state(X: torch.Tensor, W: torch.Tensor, Y: torch.Tensor, M: torch.Tensor):
    Y = torch.matmul(W, X)
    Y -= torch.matmul(M, Y)
    return torch.relu(Y)

def hebbian_learning(W: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, eta: float = 0.01):
    # Implement the Hebbian learning rule
    # Δw = learning_rate (eta) * (post_activity (g^T) * pre_activity (x))
    
    outer = torch.zeros((W.shape[0], W.shape[1]))
    for i in range(X.shape[1]):
        outer += torch.outer(Y[:, i], X[:, i])
        
    outer /= X.shape[1]

    W += eta * (outer- W)
    return W

def anti_hebbian_learning(M: torch.Tensor, Y: torch.Tensor, eta: float = 0.001):
    # Implement the Anti-Hebbian learning rule
    # ΔM = - learning_rate (eta) * (outproduct of post_activity (y*y^T))
    outer = torch.zeros((M.shape[0], M.shape[1]))
    for i in range(Y.shape[1]):
        outer += torch.outer(Y[:, i], Y[:, i])
        
    outer /= Y.shape[1]
    
    M += eta * (outer - M)
    return M

# Simulation parameters
N_LAP = 40  # Define the total number of laps
N_FRAME = 500 # Define the total frame number
N_CA3_TO_CA1 = 20

# Initial neuron states
PCs = PlaceCells(N_CA3, N_FRAME)
t = PCs.t
X = torch.from_numpy(PCs.spatial_map).float().to(device)
Y = torch.zeros((N_CA1, N_FRAME))

M = torch.zeros((N_CA1, N_CA1))
W = torch.zeros((N_CA1, N_CA3))

for i in range(N_CA3):
    W[torch.randint(0, N_CA1, size=(N_CA3_TO_CA1,1)), i] = torch.rand(size=(N_CA3_TO_CA1,1))
    
    
import matplotlib.pyplot as plt
for l in range(N_LAP):
    print(l+1)
    # Update neuron states
    Y = update_state(X, W, Y, M)
    
    # Apply learning rules
    W = hebbian_learning(W, X, Y)
    M = anti_hebbian_learning(M, Y)
    
    # Optionally: Record states and weights for analysis and visualization
    # ...
    

y = Y.cpu().numpy()
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 8))
for i in range(5):
    for j in range(5):
        axes[i, j].plot(t, y[i*5+j, :])

plt.show()

# Visualization and analysis
# ...
