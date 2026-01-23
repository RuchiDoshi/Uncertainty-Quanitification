import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd 
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np
from math import log

# --- REMOVED: Fixed random seed here so that the starting points are truly random on each run. ---

# --- 1. Dataset Generation from Gaussian Distributions ---
def generate_gaussian_data(n_samples=500):
    """
    Generates a 2D dataset sampled from three Gaussian distributions.
    
    Distribution 1: mean=[3, 3] (Peak 1)
    Distribution 2: mean=[-3, -3] (Peak 2)
    Distribution 3: mean=[-3, 3] (Peak 3)
    """
    # Distribution 1 (First Quadrant, Peak at 3, 3)
    mean1 = torch.tensor([3.0, 3.0])
    covariance1 = torch.eye(2) * 1.0**2
    dist1 = MultivariateNormal(mean1, covariance1)
    samples1 = dist1.sample((n_samples,))
    labels1 = torch.full((n_samples, 1), 0)

    # Distribution 2 (Third Quadrant, Peak at -3, -3)
    mean2 = torch.tensor([-3.0, -3.0])
    covariance2 = torch.eye(2) * (1.0)**2
    dist2 = MultivariateNormal(mean2, covariance2)
    samples2 = dist2.sample((n_samples,))
    labels2 = torch.full((n_samples, 1), 1)

    # Distribution 3 (Second Quadrant, Peak at -3, 3)
    mean3 = torch.tensor([-3.0, 3.0])
    covariance3 = torch.eye(2) * (1.0)**2
    dist3 = MultivariateNormal(mean3, covariance3)
    samples3 = dist3.sample((n_samples,))
    labels3 = torch.full((n_samples, 1), 2)

    # Combine data and labels (though only the distributions are used for GA)
    X = torch.cat([samples1, samples2, samples3], dim=0)
    y = torch.cat([labels1, labels2, labels3], dim=0).squeeze().long()
    
    return X, y, dist1, dist2, dist3

# --- 2. Multi-Layer Perceptron (MLP) Model Definition (Unused for GA) ---
class MLP(nn.Module):
    """ A simple two-layer MLP for 2D classification. Not used for this GA task. """
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(x)
        out = self.leaky_relu(out)
        out = self.output_layer(out)
        return out

# --- 3. Visualization Function (PDF and Entropy) (Unused for GA) ---
# Keeping this function for context, but it's not used in the final GA plot.
def visualize_results(X, y, model, dist1, dist2, dist3):
    # This function is not used for the current gradient ascent visualization
    pass

# --- 4. Gradient Calculation Function ---
def compute_pdf_gradient(points_x, dist1, dist2, dist3):
    """
    Calculates the gradient of the Mixture of Gaussians PDF P(x) w.r.t. the input points x.
    This function is optimized to handle a batch of points simultaneously.
    """
    # 1. Define Input: Requires gradient tracking 
    x = points_x.clone().requires_grad_(True)
    
    # 2. Forward Pass: Calculate the mixture PDF P(x)
    WEIGHT = 1.0/3.0
    prob1 = dist1.log_prob(x).exp() * WEIGHT
    prob2 = dist2.log_prob(x).exp() * WEIGHT
    prob3 = dist3.log_prob(x).exp() * WEIGHT

    # pdf_value is now a vector of PDF values, one for each point
    pdf_value = prob1 + prob2 + prob3

    # 3. Backward Pass using torch.autograd.grad (Calculates d(pdf_value)/d(x))
    gradient_tuple = torch.autograd.grad(
        outputs=pdf_value, 
        inputs=x, 
        grad_outputs=torch.ones_like(pdf_value), 
        retain_graph=False
    )

    # 4. Extract gradient and return. gradient_vector is now (N_POINTS, 2)
    gradient_vector = gradient_tuple[0]
    
    return pdf_value.detach().numpy(), gradient_vector

# --- 5. Gradient ascent algorithm and visualization for multiple points ---
def perform_and_visualize_multi_ascent(dist1, dist2, dist3):
    """
    Performs the Gradient Ascent algorithm for multiple starting points and plots the paths.
    """
    N_POINTS = 10
    
    # Setup for PDF contour plotting
    x_min, x_max = -7.0, 7.0
    y_min, y_max = -7.0, 7.0
    resolution = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    grid_points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    # Calculate the Mixture PDF over the grid
    WEIGHT = 1.0/3.0
    Z_pdf = (dist1.log_prob(grid_points).exp() * WEIGHT +
             dist2.log_prob(grid_points).exp() * WEIGHT +
             dist3.log_prob(grid_points).exp() * WEIGHT).numpy().reshape(xx.shape)

    # --- Gradient Ascent Implementation ---
    
    # Hyperparameters
    learning_rate = 5.0 
    num_steps = 20 
    
    # Initialization: 10 points uniformly distributed in [-0.5, 0.5] x [-0.5, 0.5]
    current_points = torch.rand((N_POINTS, 2)) * 5.0 - 2.5

    # Storage for paths and steps: (N_STEPS, N_POINTS, 2)
    path_points = [current_points.numpy()]
    gradient_vectors = [] 

    print("\n--- Starting Multi-Point Gradient Ascent ---")
    
    for step in range(num_steps):
        # 1. Calculate PDF values and Gradient vectors for all points
        pdf_vals, gradient_vecs = compute_pdf_gradient(current_points, dist1, dist2, dist3)
        
        # Store the gradient vectors (N_POINTS, 2)
        gradient_vectors.append(gradient_vecs.detach().numpy())
        
        # 2. Gradient Ascent Update Rule 
        update_vector = learning_rate * gradient_vecs
        
        # Move the points in the direction of the positive gradient
        current_points = current_points + update_vector
        
        # Store the new positions
        path_points.append(current_points.numpy())

        max_step_magnitude = torch.norm(update_vector, dim=1).max().item()
        # Removed detailed step printout for cleaner console
        # print(f"Step {step+1:02d}: Max Step Magnitude={max_step_magnitude:.4f}")

        if max_step_magnitude < 1e-4 and step > 5:
            print(f"All points converged after {step+1} steps.")
            break
            
    # Reshape path_points from list of (N, 2) arrays to (N_STEPS, N_POINTS, 2) numpy array
    path_points = np.array(path_points) 
    # Reshape gradient_vectors to (N_STEPS-1, N_POINTS, 2)
    gradient_vectors = np.array(gradient_vectors)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Plot the PDF Contour
    contour_pdf = ax.contourf(xx, yy, Z_pdf, levels=50, cmap='YlGnBu', alpha=0.8)
    fig.colorbar(contour_pdf, ax=ax, label='Mixture PDF Value P($\mathbf{x}$)')
    
    # 2. Plot Paths and Arrows for each point
    colors = plt.cm.get_cmap('hsv', N_POINTS)
    
    # Flags to ensure we only label the path and quiver once in the legend
    path_label_added = False
    quiver_label_added = False
    
    for i in range(N_POINTS):
        # Path
        ax.plot(path_points[:, i, 0], path_points[:, i, 1], '-', 
                linewidth=1.5, color=colors(i), alpha=0.8,
                label='Ascent Path' if not path_label_added else "")
        if not path_label_added: path_label_added = True
        
        # Start and End Points (No individual labels, color separates them)
        ax.plot(path_points[0, i, 0], path_points[0, i, 1], 'o', 
                markersize=8, color=colors(i), markeredgecolor='k')
        ax.plot(path_points[-1, i, 0], path_points[-1, i, 1], 'X', 
                markersize=10, color=colors(i), markeredgecolor='k')
                
        # Gradient Arrows (Quivers)
        U = gradient_vectors[:, i, 0] * learning_rate 
        V = gradient_vectors[:, i, 1] * learning_rate
        X_start = path_points[:-1, i, 0]
        Y_start = path_points[:-1, i, 1]

        ax.quiver(X_start, Y_start, U, V, 
                  color=colors(i), 
                  angles='xy', scale_units='xy', scale=1.0, 
                  width=0.003, headwidth=5, headlength=7, alpha=0.6,
                  label='Step Vector $\eta \cdot \nabla P(\mathbf{x})$' if not quiver_label_added else "")
        if not quiver_label_added: quiver_label_added = True


    # Plot the true peak locations (These are the actual labels for the convergence points)
    ax.plot(3.0, 3.0, 'k*', markersize=15, label='Peak 1 (3, 3)')
    ax.plot(-3.0, -3.0, 'k*', markersize=15, label='Peak 2 (-3, -3)')
    ax.plot(-3.0, 3.0, 'k*', markersize=15, label='Peak 3 (-3, 3)')

    ax.set_title('Multi-Point Gradient Ascent on a Mixture of Gaussians PDF')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.axis('equal') 
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    # Generate Data and Distribution Objects
    X, y, dist1, dist2, dist3 = generate_gaussian_data(n_samples=500)
    print("Gaussian distributions defined.")

    # --- Multi-Point Gradient Ascent Simulation ---
    perform_and_visualize_multi_ascent(dist1, dist2, dist3)