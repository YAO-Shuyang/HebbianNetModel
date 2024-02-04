import torch

def update_state(X: torch.Tensor, W: torch.Tensor, Y: torch.Tensor, M: torch.Tensor, thre: float | torch.Tensor) -> torch.Tensor:
    """update_state: Update the status of neurons

    Parameters
    ----------
    X : torch.Tensor
        Input from CA3, typically from CA3 place cells with single fields but varying in sizes.
    W : torch.Tensor
        The weighted matrix of connected neuron.
    Y : torch.Tensor
        Output from CA1
    M : torch.Tensor
        The weighted matrix of connected lateral inhibitory connections.
    """
    Y = torch.matmul(W, X)
    Y -= torch.matmul(M, Y)
    Y = torch.relu(Y-thre)
    #for i in range(Y.shape[0]):
    #    Y[i, :] = Y[i, :]/torch.max(Y[i, :])
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