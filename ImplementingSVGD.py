import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd 
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np
from math import log
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.cm as cm

class SVGD:
    def __init__(self):
        # This class encapsulates the SVGD logic for external use
        pass

    # --- 1. Dataset Generation from Gaussian Distributions ---
    def generate_gaussian_data(self, n_samples=500):
        """
        Generates a 2D dataset sampled from three Gaussian distributions.
        
        Distribution 1: mean=[3, 3] (Peak 1)
        Distribution 2: mean=[-3, -3] (Peak 2)
        Distribution 3: mean=[-3, 3] (Peak 3)
        """
        mean1 = torch.tensor([3.0, 3.0])
        covariance1 = torch.eye(2) * 1.0**2
        dist1 = MultivariateNormal(mean1, covariance1)

        mean2 = torch.tensor([-3.0, -3.0])
        covariance2 = torch.eye(2) * (1.0)**2
        dist2 = MultivariateNormal(mean2, covariance2)

        mean3 = torch.tensor([-3.0, 3.0])
        covariance3 = torch.eye(2) * (1.0)**2
        dist3 = MultivariateNormal(mean3, covariance3)
        
        # We only need the distribution objects for the PDF calculation
        return dist1, dist2, dist3

    # --- New: Initialize Points from different distributions ---
    def initialize_points(self, N_POINTS, mode='uniform'):
        """Initializes particle points based on the selected mode."""
        
        # Ensures different random points each time the function is called
        # uniform: randomly, with equal probability) across a small square region
        # otherwise gaussian: points initialized in tight cluster around central point (0, 0) according to a Gaussian distribution.
        
        if mode == 'uniform':
            # Uniformly distributed in [-0.5, 0.5] x [-0.5, 0.5]
            # This gives a broad exploration start
            initial_points = torch.rand((N_POINTS, 2)) * 1.0 - 0.5 #*1 = scale
            print(f"Initialized {N_POINTS} particles uniformly.")
            
        elif mode == 'gaussian':
            # Tightly clustered around the origin (0, 0)
            # This tests if the repulsion force can push them away from a single peak
            mean = torch.tensor([0.0, 0.0])
            covariance = torch.eye(2) * 0.1**2 # torch.eye(2): 2 by 2 identity matrix, multiplied by 0.1^2: varaiance/sd squared
            gaussian_dist = MultivariateNormal(mean, covariance) # creates gaussian dist w specified params
            initial_points = gaussian_dist.sample((N_POINTS,))
            print(f"Initialized {N_POINTS} particles in a tight Gaussian cluster around (0, 0).")
            
        else:
            raise ValueError("Invalid initialization mode.")
            
        return initial_points

    # --- 2. Median Heuristic for RBF Kernel Bandwidth (sigma) ---
    @staticmethod
    def median_heuristic(X):
        """
        paramter X: a pytorch tensor that hold the coordinates of the particles 
        Calculates the RBF kernel bandwidth sigma using the median of the pairwise 
        squared distances, scaled by log(N).
        sets the "repulsion distance" to ensure that the points dont all clump on one peak
        """
        N = X.size(0)
        
        # Compute the pairwise squared Euclidean distances (N x N matrix)
        dist_sq = torch.cdist(X, X, p=2).pow(2) 
        """
        cdist: computed distance (calculates distance between every pair of vectors in the 2 input tensors)
        since both input tensors are X, it calculates the distance from every particle to every other particle 
        (including itself).
        p=2: specifies distance metric -> p=2 is the Euclidean distance
        returns matrix of Euclidean distances
        squares each element bc Euclidean distance involves a square root which slows down opperations
        """
        # Select the upper triangle (excluding diagonal) to get unique pairs
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool() #ignores distances that are 0 and duplicates (a to b equals b to a)
        
        if mask.sum() == 0:
            # Not enough points, return a safe default
            return 1.0 
        

        #this is a list
        distances_for_median = dist_sq[mask] #uses the boolean matrix mask to only collect the non zero/same vals
        
        
        # Find the median of the squared distances
        med_sq = torch.median(distances_for_median)
        
        # SVGD often uses this scaling factor for robustness
        scaling_factor = torch.log(torch.tensor(N).float()) #calculates scaling factor by ln(N/numvals)
        if scaling_factor == 0: scaling_factor = 1.0 #scaling factor will be a denomiator later and therefore cant be 0, so it is 1 if 0
        
        # Calculate sigma = sqrt(med^2 / log(N))
        """"
        This is the RBF kernel bandwith (the radius of influence that dictates how strongly two
        particles repel each other based on their distance.)
        """
        sigma_sq = med_sq / scaling_factor
        sigma = torch.sqrt(sigma_sq).item() #.item() extracts the float num from the pytorch tensor
        
        return sigma

    # --- 3. Log-PDF Gradient Calculation (The "Attractive" Force) ---
    @staticmethod
    def compute_log_pdf_and_gradient(points_x, dist1, dist2, dist3):
        """
        Calculates the log-PDF value log P(x) and its gradient d log P(x) / dx.
        points_x: the rand starting points we generated
        distX: target distributions (sd and mean)
        """
        # Requires gradient tracking for the backward pass
        x = points_x.clone().requires_grad_(True) #clones the particles and requires_grad tracks the operations done to it
        WEIGHT = 1.0/3.0 #sets even weight to each distribution
        
        # Calculate PDF: P(x) = W1*P1 + W2*P2 + W3*P3
        prob1 = dist1.log_prob(x).exp() * WEIGHT
        prob2 = dist2.log_prob(x).exp() * WEIGHT
        prob3 = dist3.log_prob(x).exp() * WEIGHT
        
        pdf_value = prob1 + prob2 + prob3
        
        # FIX: Add a small epsilon for numerical stability before taking the log
        epsilon = 1e-8 #to guarantee that the log is a number and not -inf
        log_pdf_value = torch.log(pdf_value + epsilon) 

        # Backward pass for the gradient of log_pdf_value w.r.t x
        #triggers automatic differntiation to calculate attractice force
        #CONFUSED ABT THIS
        gradient_tuple = torch.autograd.grad(
            outputs=log_pdf_value, 
            inputs=x, 
            grad_outputs=torch.ones_like(log_pdf_value), 
            retain_graph=False
        )

        #a tuple is always returned from prev func bc multipe inputs could be passed, but we only need the first tensor
        #(0) so we are extracting that from the returned tuple
        gradient_vector = gradient_tuple[0]
        
        #returns two components defining the attractive force, (the gradient) and the current state score (the log-PDF value)
        return log_pdf_value.detach().numpy(), gradient_vector

    # --- 4. RBF Kernel and its Gradient (The "Repulsive" Force) ---
    @staticmethod
    def rbf_kernel_and_grad(X, sigma):
        """
        Calculates the RBF kernel matrix K(X, X) and the sum of its analytical gradient 
        dK(x_j, x_i) / dx_i (Term 2 in SVGD).

        uses bandwidth to calculate the actual repulsive forces
        """
        #N: num points D:dimensions(2)
        N, D = X.size()
        
        # 1. Pairwise squared distances: ||x_i - x_j||^2
        X_i = X.unsqueeze(1) # (N, 1, D)
        X_j = X.unsqueeze(0) # (1, N, D)
        dist_sq = torch.sum((X_i - X_j)**2, dim=2) #gets squared Euclidean distances ∥xi​−xj​∥.
        
        # 2. Kernel Matrix K(X, X)
        #computes RBF kernel matrix - N x N matrix where each value represents the influence between each particle on another
        K = torch.exp(-dist_sq / (2.0 * sigma**2)) # (N, N)
        
        # 3. Analytical Gradient Sum (for the repulsion term)
        
        # X_diff[i, j, :] = (x_j - x_i)
        #calculates difference vector for all NxN pairs shape(N,N,D)
        X_diff = X_j - X_i # (N, N, D)
        
        # K_broadcast is K(x_i, x_j) (N, N, 1)
        #Reshapes K from (N,N) to (N,N,1). This allows K(xi​,xj​) to be broadcasted across the D dimensions
        K_broadcast = K.unsqueeze(2) 
        
        # dK/dx_i (N, N, D)
        #The result is a vector that points in the direction of xj​−xi​ (from i to j), but its magnitude is scaled by the kernel value.
        grad_K_i = K_broadcast * X_diff / (sigma**2)
        
        # Term 2 in SVGD: Sum over j of the gradient of K w.r.t. x_i (dK(x_j, x_i)/dx_i).
        sum_grad_K = torch.sum(grad_K_i, dim=1) # (N, D)
        
        # Return K(N, N) and the sum of the gradient of K w.r.t X (N, D)
        return K.detach(), sum_grad_K.detach()

    # --- 5. SVGD algorithm and data collection ---
    def run_svgd(self, initial_points, dist1, dist2, dist3, sigma_mode='median', num_steps=30, learning_rate=5.0):
        """
        Performs the SVGD algorithm and returns the path, forces, and parameters.
        """
        current_points = initial_points.clone()
        N_POINTS = current_points.size(0)
        
        # Determine sigma either using the function (dynamically) or set to 1.0
        if sigma_mode == 'median':
            sigma = self.median_heuristic(current_points)
        else: 
            sigma = 1.0

        print(f"\n--- Running SVGD with {len(initial_points)} particles, sigma={sigma:.2f} ---")

        # Storage
        #creates a list to store positions of all particles at each step, starting w initial locations - .numpy() converts PyTorch tensor to NumPy array for better sotorage
        path_points = [current_points.numpy()]
        #list to store combined force (attractive and repulsive) and its components for vizualizations
        total_gradient_vectors = [] 
        attractive_force_vectors = []
        repulsive_force_vectors = []

        #runs loops for specifed number of steps
        for step in range(num_steps):
            
            # 1. Attractive Force Component
            _, log_pdf_grad = self.compute_log_pdf_and_gradient(current_points, dist1, dist2, dist3)
            
            # 2. Repulsive Force Component
            K, sum_grad_K = self.rbf_kernel_and_grad(current_points, sigma)
            
            # Calculate Attractive Force: K * grad_log_P (Term 1)
            term1 = torch.matmul(K, log_pdf_grad) # (N, D)
            
            # Calculate Repulsive Force: Sum_j grad_K (Term 2)
            term2 = sum_grad_K # (N, D)
            
            # SVGD Update Field phi(x_i) = (1/N) * sum_j [Term 1 + Term 2]
            #directional vector of where point should go
            phi = (term1 + term2) / N_POINTS # (N, D)
            
            # Store components for visualization (only the first step for simplicity)
            if step == 0:
                attractive_force_vectors.append(term1.detach().numpy() / N_POINTS) # Normalized
                repulsive_force_vectors.append(term2.detach().numpy() / N_POINTS) # Normalized

            total_gradient_vectors.append(phi.detach().numpy())
            
            # 3. Apply the update
            update_vector = learning_rate * phi.detach()
            current_points = current_points.detach() + update_vector
            
            # Store the new positions
            path_points.append(current_points.numpy())

            max_step_magnitude = torch.norm(update_vector, dim=1).max().item()
            
            #if the max distacne any particle moved in given step is below 10^-4 assumes convergence and breaks loop early
            if max_step_magnitude < 1e-4 and step > 5:
                print(f"Convergence after {step+1} steps.")
                break
                
        # Scale force vectors by learning rate for plotting as velocity
        attractive_force_vectors = np.array(attractive_force_vectors) * learning_rate
        repulsive_force_vectors = np.array(repulsive_force_vectors) * learning_rate
                
        return np.array(path_points), np.array(total_gradient_vectors), attractive_force_vectors, repulsive_force_vectors, sigma

    # --- 6. Plotting Functions ---

    def plot_density_contours(self, ax, dist1, dist2, dist3):
        """Plots the background density contours."""
        x_min, x_max = -7.0, 7.0
        y_min, y_max = -7.0, 7.0
        resolution = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                             np.arange(y_min, y_max, resolution))
        grid_points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

        WEIGHT = 1.0/3.0
        Z_pdf = (dist1.log_prob(grid_points).exp() * WEIGHT +
                 dist2.log_prob(grid_points).exp() * WEIGHT +
                 dist3.log_prob(grid_points).exp() * WEIGHT).numpy().reshape(xx.shape)

        contour_pdf = ax.contourf(xx, yy, Z_pdf, levels=50, cmap='YlGnBu', alpha=0.8)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.axis('equal') 
        
        # Plot the true peak locations
        ax.plot(3.0, 3.0, 'k*', markersize=15, label='Peak 1 (3, 3)')
        ax.plot(-3.0, -3.0, 'k*', markersize=15, label='Peak 2 (-3, -3)')
        ax.plot(-3.0, 3.0, 'k*', markersize=15, label='Peak 3 (-3, 3)')
        return contour_pdf

    def plot_paths(self, path_points, total_gradient_vectors, sigma, title_suffix, N_POINTS, learning_rate, dist1, dist2, dist3):
        """Plots the paths and total velocity for a full SVGD run. (Fixed signature)"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        # PASS DISTRIBUTIONS HERE
        contour_pdf = self.plot_density_contours(ax, dist1, dist2, dist3) 
        fig.colorbar(contour_pdf, ax=ax, label='Mixture PDF Value P($\mathbf{x}$)')
        
        colors = plt.cm.get_cmap('hsv', N_POINTS)
        path_label_added = False
        
        for i in range(N_POINTS):
            # Path
            ax.plot(path_points[:, i, 0], path_points[:, i, 1], '-', 
                    linewidth=1.5, color=colors(i), alpha=0.8,
                    label='Ascent Path' if not path_label_added else "")
            if not path_label_added: path_label_added = True
            
            # Start and End Points
            ax.plot(path_points[0, i, 0], path_points[0, i, 1], 'o', 
                    markersize=8, color=colors(i), markeredgecolor='k', label='Start' if i == 0 else "")
            ax.plot(path_points[-1, i, 0], path_points[-1, i, 1], 'X', 
                    markersize=10, color=colors(i), markeredgecolor='k', label='End' if i == 0 else "")
                    
            # Total SVGD Velocity Arrows (Quivers)
            U = total_gradient_vectors[:, i, 0] * learning_rate 
            V = total_gradient_vectors[:, i, 1] * learning_rate
            X_start = path_points[:-1, i, 0]
            Y_start = path_points[:-1, i, 1]

            ax.quiver(X_start, Y_start, U, V, 
                      color=colors(i), 
                      angles='xy', scale_units='xy', scale=1.0, 
                      width=0.003, headwidth=5, headlength=7, alpha=0.6,
                      label='SVGD Velocity $\eta \cdot \phi(\mathbf{x})$' if i == 0 else "")

        ax.set_title(f'SVGD on Mixture of Gaussians: {title_suffix} ($\sigma$={sigma:.2f})')
        ax.legend()
        plt.show()

    def plot_force_decomposition(self, initial_points, attractive_forces, repulsive_forces, sigma, N_POINTS, dist1, dist2, dist3):
        """Plots the attractive and repulsive force components at the first step. (Fixed signature)"""
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        colors = plt.cm.get_cmap('hsv', N_POINTS)

        # --- Left Plot: Attractive Force (Density Gradient) ---
        ax_attraction = axes[0]
        # PASS DISTRIBUTIONS HERE
        self.plot_density_contours(ax_attraction, dist1, dist2, dist3)
        # Changed from f-string to normal string to avoid NameError with \mathbf{x}
        ax_attraction.set_title('1. Attractive Force (Density Gradient) $\\eta \\cdot K \\nabla \\log P(\\mathbf{x})$')
        
        for i in range(N_POINTS):
            U_attract = attractive_forces[0, i, 0] 
            V_attract = attractive_forces[0, i, 1]
            
            # Plot initial points
            ax_attraction.plot(initial_points[i, 0], initial_points[i, 1], 'o', 
                               markersize=8, color=colors(i), markeredgecolor='k')
                               
            # Plot the attractive force arrow
            ax_attraction.quiver(initial_points[i, 0], initial_points[i, 1], U_attract, V_attract, 
                                 color=colors(i), 
                                 angles='xy', scale_units='xy', scale=1.0, 
                                 width=0.005, headwidth=5, headlength=7, alpha=0.9)
        ax_attraction.legend()

        # --- Right Plot: Repulsive Force (Kernel) ---
        ax_repulsion = axes[1]
        # PASS DISTRIBUTIONS HERE
        self.plot_density_contours(ax_repulsion, dist1, dist2, dist3)
        # Changed from f-string to normal string to avoid NameError with \mathbf{x}
        ax_repulsion.set_title('2. Repulsive Force (Kernel) $\\eta \\cdot \\nabla K(\\mathbf{x})$')
        
        for i in range(N_POINTS):
            U_repulse = repulsive_forces[0, i, 0] 
            V_repulse = repulsive_forces[0, i, 1]
            
            # Plot initial points
            ax_repulsion.plot(initial_points[i, 0], initial_points[i, 1], 'o', 
                               markersize=8, color=colors(i), markeredgecolor='k')
                               
            # Plot the repulsive force arrow
            ax_repulsion.quiver(initial_points[i, 0], initial_points[i, 1], U_repulse, V_repulse, 
                                 color=colors(i), 
                                 angles='xy', scale_units='xy', scale=1.0, 
                                 width=0.005, headwidth=5, headlength=7, alpha=0.9)
        ax_repulsion.legend()
        
        fig.suptitle(f'SVGD Force Decomposition at Start (Median Heuristic, $\sigma$={sigma:.2f})', fontsize=16)
        plt.show()

    def plot_3d_particle_density(self, final_points, sigma, dist1, dist2, dist3):
        """
        Plots the 2D histogram of final particle positions in 3D space 
        (z-axis = density/count) and overlays the scaled target PDF surface.
        """
        fig = plt.figure(figsize=(12, 10))
        # Add a 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        x_min, x_max = -7.0, 7.0
        y_min, y_max = -7.0, 7.0
        N_BINS = 15 # Optimized for N_POINTS=10

        # 1. Calculate 2D Histogram (Counts)
        H, xedges, yedges = np.histogram2d(
            final_points[:, 0], 
            final_points[:, 1], 
            bins=N_BINS, 
            range=[[x_min, x_max], [y_min, y_max]]
        )

        # 2. Prepare grid for histogram surface plotting (using bin centers)
        x_mid = (xedges[:-1] + xedges[1:]) / 2
        y_mid = (yedges[:-1] + yedges[1:]) / 2
        X_hist, Y_hist = np.meshgrid(x_mid, y_mid)
        Z_hist = H.T # Transpose is needed to match array dimensions
        
        # 3. Plot the histogram surface
        ax.plot_surface(X_hist, Y_hist, Z_hist, cmap=cm.viridis, alpha=0.8, 
                        edgecolor='k', linewidth=0.5, rcount=N_BINS, ccount=N_BINS)

        # 4. Plot True PDF Surface (Target distribution) for visual comparison
        resolution = 0.2
        x_fine = np.arange(x_min, x_max, resolution)
        y_fine = np.arange(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x_fine, y_fine)
        grid_points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

        WEIGHT = 1.0/3.0
        Z_pdf = (dist1.log_prob(grid_points).exp() * WEIGHT +
                 dist2.log_prob(grid_points).exp() * WEIGHT +
                 dist3.log_prob(grid_points).exp() * WEIGHT).numpy().reshape(xx.shape)
        
        if H.max() > 0 and Z_pdf.max() > 0:
            Z_pdf_scaled = Z_pdf * (H.max() / Z_pdf.max()) * 0.8
        else:
            Z_pdf_scaled = Z_pdf * 1.0 

        ax.plot_surface(xx, yy, Z_pdf_scaled, cmap=cm.plasma, alpha=0.3)

        # 5. Final Touches
        ax.set_xlabel('$x_1$ position', fontsize=12)
        ax.set_ylabel('$x_2$ position', fontsize=12)
        ax.set_zlabel('Particle Density (Count)', fontsize=12)
        ax.set_zlim(0, Z_hist.max() * 1.5)
        
        ax.set_title(f'3D Density of Converged SVGD Particles vs. Target PDF ($\sigma$={sigma:.2f})', fontsize=14)
        
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='green', lw=4, alpha=0.8),
                        Line2D([0], [0], color='magenta', lw=4, alpha=0.3)]
        ax.legend(custom_lines, ['Particle Density (Histogram)', 'Target PDF (Mixture)'], loc='upper left')

        plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create the SVGD object
    svgd_engine = SVGD()
    
    # Generate Distribution Objects
    dist1, dist2, dist3 = svgd_engine.generate_gaussian_data()
    N_POINTS = 10
    LEARNING_RATE = 5.0
    
    print("Gaussian distributions defined.")

    # SCENARIO 1: Uniform Initialization (Median Heuristic)
    initial_points_uniform = svgd_engine.initialize_points(N_POINTS, mode='uniform')
    
    path_uniform, total_grad_uniform, attractive_forces, repulsive_forces, sigma_uniform = svgd_engine.run_svgd(
        initial_points_uniform, dist1, dist2, dist3, sigma_mode='median', learning_rate=LEARNING_RATE
    )
    
    svgd_engine.plot_force_decomposition(initial_points_uniform.numpy(), attractive_forces, repulsive_forces, sigma_uniform, N_POINTS, dist1, dist2, dist3)
    
    svgd_engine.plot_paths(path_uniform, total_grad_uniform, sigma_uniform, 
                           'Median Heuristic (Uniform Start)', N_POINTS, LEARNING_RATE, dist1, dist2, dist3)

    final_points_uniform = path_uniform[-1, :, :] 
    svgd_engine.plot_3d_particle_density(final_points_uniform, sigma_uniform, dist1, dist2, dist3)