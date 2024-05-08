import torch 
def mixt(alpha=1.0, K=7, **kwargs):
    """ 
    Samples one observation from a DPM model. 

    -Parameters:
    - alpha: the DP concentration parameter
    - K: the truncation for the number of components
    """
    w = torch.distributions.Dirichlet(torch.full((K,), alpha/K)).sample()
    if kwargs: #if trajectory isn't new
        return torch.distributions.Normal(**kwargs).sample()
    else: #if trajectory is new
        sigma2 = torch.distributions.InverseGamma(1.0, 1.0).sample()
        loc= torch.distributions.Normal(0.0, 1.0).sample()
        return torch.distributions.Normal(loc, torch.sqrt(sigma2)).sample().unsqueeze(0)
