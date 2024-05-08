import torch

def logreg(x, mu, sigma):
    """
    Samples from a logistic regression based on scalar x.

    Parameters:
    - x (float or torch.Tensor): the covariate scalar.
    - mu (float): the mean for the beta.
    - sigma (float): the variance for the beta.

    Returns:
    - y (int): Sample {-1, 1} based on logistic regression.
    """
    # Sample beta from a normal distribution
    beta = torch.normal(mean=mu, std=sigma**0.5)

    # Compute probability using logistic function
    p = 1.0 / (1.0 + torch.exp(-x * beta))
    
    # Sample binary outcome based on probability
    y = torch.distributions.Bernoulli(p).sample()
    y = int(y * 2 - 1)  # Convert {0, 1} to {-1, 1}

    return y
