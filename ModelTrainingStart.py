import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. THE PARTICLE (Neural Network Architecture) ---
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__() #arguments self and regressionNet not necessary here - occasionally useful for backwards compatability
        """
        establishes the type of layers that the neural net will contain
        self.layers is just the varaible name for a varaible storing a sequence of layers
        layer type:sequential
        """
        self.layers = nn.Sequential(  
            nn.Linear(1, 16), #1,16: input sample size: 1 - output sample size: 16
            nn.ReLU(),
            nn.Linear(16, 1)   # too small (second order function)
        )

    def forward(self, x):
        return self.layers(x)

# --- 2. THE MATH ENGINE (Your original SVGD logic) ---
class SVGD:
    @staticmethod
    def median_heuristic(X):
        N = X.size(0)
        dist_sq = torch.cdist(X, X, p=2).pow(2)
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        if mask.sum() == 0: return 1.0
        
        med_sq = torch.median(dist_sq[mask])  # you don't have to get upper triangular mask for this. Median of distances is fine
        scaling_factor = torch.log(torch.tensor(N).float())
        if scaling_factor == 0: scaling_factor = 1.0

        # double check but I am sure median heuristic requires 2.0 * log(N) # would double check RBF median heuristic kernel
        
        return torch.sqrt(med_sq / scaling_factor).item()

    @staticmethod
    def rbf_kernel_and_grad(X, sigma):
        N, D = X.size()
        X_i = X.unsqueeze(1)
        X_j = X.unsqueeze(0)
        dist_sq = torch.sum((X_i - X_j)**2, dim=2)
        
        K = torch.exp(-dist_sq / (2.0 * sigma**2))  # above comment most likely because of the second factor
        X_diff = X_j - X_i
        grad_K_i = K.unsqueeze(2) * X_diff / (sigma**2) # e^(-d/(2s^2)) # 2*s^2 not just s^2 [might be wrong]
        sum_grad_K = torch.sum(grad_K_i, dim=1)
        
        return K.detach(), sum_grad_K.detach()

# --- 3. THE TRAINER (The "Body" that uses the engine) ---
class SVGDRegressionTrainer:
    def __init__(self, n_particles=10):
        self.n_particles = n_particles
        self.models = [RegressionNet() for _ in range(n_particles)]
        self.svgd = SVGD()

    def get_theta(self):
        """Helper: Flattens all model weights into an (N, D) matrix."""
        all_params = []
        for model in self.models:
            p_vector = torch.cat([p.data.view(-1) for p in model.parameters()])
            all_params.append(p_vector)
        return torch.stack(all_params) 

    def update_models(self, theta_new):
        """Helper: Puts updated weights back into the models."""
        for i, model in enumerate(self.models):
            pointer = 0
            for p in model.parameters():
                num = p.numel()
                p.data.copy_(theta_new[i, pointer:pointer + num].view(p.shape))
                pointer += num

    def compute_loss_gradients(self, X, y):
        """Calculates the 'Attractive Force' (Log-Likelihood Gradient)."""
        gradients = []
        total_loss = 0
        for model in self.models:
            model.zero_grad()
            pred = model(X)
            loss = nn.MSELoss()(pred, y)
            total_loss += loss.item()
            # Negative loss acts as log-likelihood for attraction
            (-loss).backward()
            
            g_vector = torch.cat([p.grad.view(-1) for p in model.parameters()])
            gradients.append(g_vector)
        
        avg_loss = total_loss / self.n_particles
        return torch.stack(gradients), avg_loss

    def train_step(self, X, y, lr=0.001):
        # 1. Get current weights as particles
        theta = self.get_theta()
        
        # 2. Get Attractive Force (Loss Gradients) AND the loss value
        ln_p_grad, current_loss = self.compute_loss_gradients(X, y)
        
        # 3. Get Repulsive Force (Kernel logic)
        sigma = self.svgd.median_heuristic(theta)
        K, sum_grad_K = self.svgd.rbf_kernel_and_grad(theta, sigma)
        
        # 4. SVGD Update Math
        phi = (torch.matmul(K, ln_p_grad) + sum_grad_K) / self.n_particles
        
        # 5. Apply update and push back to models
        theta_new = theta + lr * phi
        self.update_models(theta_new)
        
        return current_loss # Now 'current_loss' is defined and safe to return
    
# --- 4. DATA LOADING & EXECUTION ---
if __name__ == "__main__":
    # Load and Filter Data
    df = pd.read_csv('2.2_City_Daily_NoMissing.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Cleaning
    df['Ct_Value'] = pd.to_numeric(df['Ct_Value'].replace('NA', np.nan), errors='coerce')
    df['NewCases'] = pd.to_numeric(df['NewCases'], errors='coerce')
    df = df.dropna(subset=['NewCases', 'Ct_Value'])

    # Scale for stability
    X = torch.tensor((df['NewCases'].values - df['NewCases'].min()) / 
                     (df['NewCases'].max() - df['NewCases'].min()), dtype=torch.float32).view(-1, 1)
    y = torch.tensor((df['Ct_Value'].values - df['Ct_Value'].min()) / 
                     (df['Ct_Value'].max() - df['Ct_Value'].min()), dtype=torch.float32).view(-1, 1)

    # Initialize and Train
    trainer = SVGDRegressionTrainer(n_particles=10)
    print("Starting training on COVID-19 data...")
    
    for epoch in range(100):
        trainer.train_step(X, y, lr=0.01)
        if epoch % 20 == 0:
            print(f"Epoch {epoch} completed.")

    print("Training finished. You now have 10 different models representing uncertainty.")
