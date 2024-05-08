import torch
import pred_coreset


class Discrepancy():
    def __init__(self, disc, grad_disc = None):
        """
        Initialize the simulator with parameters.

        Parameters:
        - disc: the discrepancy function between two data points
        - grad_disc: (optional) gradient of the discrepancy function
        """

        self.disc = disc
        self.grad_disc = grad_disc

    def optimal_w(self, full_traj, corepred, N = 100):
        """
        Gets optiomal w for a given sample from a Predictor object corepred.

        Parameters:
        - full_traj: the tensor with the trajectory from the full dataset
        - corepred: the Predictor object based on the coreset support
        - N: size of the trajectory to sample
        """
        if self.grad_disc is None:
            # If no gradient use ADAM
            w = torch.ones(corepred.N, requires_grad=True) # w needs to be optimized

            # The optimization loop
            optimizer = torch.optim.Adam([w], lr=0.01)  # Using Adam optimizer
            for step in range(1000):  # Number of optimization steps
                optimizer.zero_grad()  # Clear previous gradients
                x_w = corepred.sample_traj(N, w=w)  # Compute x_w using the function P
                # if not full_traj.size == x_w.size:
                #    raise ValueError("The lengths of the sampled trajectories are not equal")
            
                loss = self.disc(x_w, full_traj)  # Compute the l2 wasserstein distance
                loss.backward()  # Compute gradients through automatic differentiation
                optimizer.step()  # Update w based on gradients

            #if step % 100 == 0:  # Print the loss every 100 steps
            #   print(f'Step {step}, Loss: {loss.item()}')
            return  w
        
        else:
            w = torch.ones(corepred.N, requires_grad=False)

            # Create an optimizer using Stochastic Gradient Descent with momentum
            optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)

            # Optimization loop
            for step in range(1000):  # Number of optimization steps
                optimizer.zero_grad()  # Clear any residual gradients
                x_w = corepred.sample_traj(N, w=w)

                if x_w.size() != full_traj.size():
                    raise ValueError("The lengths of the sampled trajectories are not equal")

                # Compute the loss function
                loss = self.disc(x_w, full_traj)

                # Compute the gradients using the explicit `grad_disc` function
                grad = self.grad_disc(w, full_traj, x_w)

                # Assign the computed gradients to `w.grad`
                w.grad = grad

                # Update `w` based on gradients using SGD
                optimizer.step()

                if step % 100 == 0:  # Print the loss every 100 steps
                    print(f'Step {step}, Loss: {loss.item()}')

            return w

