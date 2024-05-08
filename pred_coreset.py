import numpy as np
#import ot 
from scipy import stats
import torch
from multiprocessing import Pool
from tqdm import tqdm

class Predictor():
    def __init__(self, x, alpha, dist, lambd):
        """
        Initialize the simulator with parameters.

        Parameters:
        - x: the observed data points. Dim=1 is the indicator of point
        - alpha: positive float, alpha parameter
        - dist: probability distribution (scipy.stats)
        - lambd: sparsity parameter
        """
        if alpha <= 0:
            raise ValueError("Alpha must be a positive float.")
        self.x = x
        self.N = len(x)
        self.alpha = alpha
        self.dist = dist
        self.lambd = lambd



    def sample_traj(self, size, w=None, **kwargs):
        """
        Simulate from the posterior predictive of the DP. Only one layer hierarchy supported so far

        Parameters:
        - size: integer, number of data points to generate
        - w: the weights of the coreset

        Returns:
        - data: tensor array containing generated data points
        """
        traj = torch.zeros(size, requires_grad=False)


        bag = w * self.x if w is not None else self.x
        #bag = bag.detach() 
        # Choose if doing bootstrap or sample new point
        for i in range(traj.size(dim=0)):
            if np.random.random() < self.alpha /(self.alpha + self.N):
                # **kwargs automatically passes the hyperparameters if there's any
                traj[i] = self.dist(**kwargs) 
            else:
                traj[i] = bag[torch.randint(0, len(bag), (1,))]
                bag = torch.cat((bag, traj[i].unsqueeze(0)))
        # Generate data points from the probability density function f
        return torch.sort(traj)[0]
    
   

        
def sample_coreset(full_traj, corepred):
    """
    Get one sample of w. The subset we need to weight needs to be preselected
    
    Parameters: 
    - full_traj: a torch tensor with a sampled trajectory from the full dataset
    - corepred: an object of type Predictor with the reduced dataset
    """ 
    # Assuming x and w are 1D tensors. 
    w = torch.ones(corepred.N, requires_grad=True) # w needs to be optimized

    # The optimization loop
    optimizer = torch.optim.Adam([w], lr=0.01)  # Using Adam optimizer
    for step in range(1000):  # Number of optimization steps
        optimizer.zero_grad()  # Clear previous gradients
        x_w = corepred.sample_traj(100, w=w)  # Compute x_w using the function P
       # if not full_traj.size == x_w.size:
        #    raise ValueError("The lengths of the sampled trajectories are not equal")
        
        loss = torch.norm(torch.sort(x_w)[0] - full_traj, p=2)  # Compute the l2 wasserstein distance
        loss.backward()  # Compute gradients through automatic differentiation
        optimizer.step()  # Update w based on gradients

        #if step % 100 == 0:  # Print the loss every 100 steps
         #   print(f'Step {step}, Loss: {loss.item()}')
    return  w

def _coreset_map(t, x, core_x, alpha, dist, lambd, discrepancy):
    """
    Aux function that performs the coreset sampling once
    """
    full_pred = Predictor(x, alpha, dist, lambd)
    core_pred = Predictor(core_x, alpha, dist, lambd)
    traj = full_pred.sample_traj(100)  # Get a new instance of Pt
    return discrepancy.optimal_w(traj, core_pred)  # Get a coreset for Pt


def get_coreset(x, B, core_x, alpha, dist, lambd, discrepancy, parallel=False):
    """
    Main function given data x we obtain a predictive coreset

    - Parameters: 
    - x: 1-D numpy array with the full data vector
    - B: the number of bootstrap samples wanted
    - core_x: the coreset support
    - alpha: the concentration for the DP prior
    - dist: the model we sample from
    - lambd: sparsity parameter
    - discrepancy: Discrepancy object
    - parallel: multicore yes or not
    """
        
    # Initialize matrix to store samples
    W = torch.ones((len(core_x), B))

    if parallel:
        # Use multiprocessing
        with Pool() as pool:
            args = [(t, x, core_x, alpha, dist, lambd) for t in range(B)]
            results = list(tqdm(pool.starmap(_coreset_map, args), total=B))
    else:
        # Use a sequential and vectorized approach
        full_pred = Predictor(x, alpha, dist, lambd)
        core_pred = Predictor(core_x, alpha, dist, lambd)
        results = []
        for t in tqdm(range(B)):
            traj = full_pred.sample_traj(100)
            results.append(discrepancy.optimal_w(traj, core_pred))

    # Fill the results into W
    for i, result in enumerate(results):
        W[:, i] = result

    return W
