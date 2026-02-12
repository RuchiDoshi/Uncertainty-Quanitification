import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Regression netural network layers
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__() 
        self.layers = nn.Sequential(  
            nn.Linear(1, 16), #16 neurons
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

# SVGD
class SVGD:
    @staticmethod
    def median_heuristic(X):
        N = X.size(0)
        dist_sq = torch.cdist(X, X, p=2).pow(2)
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        if mask.sum() == 0: return 1.0
       
        med_sq = torch.median(dist_sq[mask])
        scaling_factor = torch.log(torch.tensor(N).float())
        if scaling_factor == 0: scaling_factor = 1.0
       
        return torch.sqrt(med_sq / scaling_factor).item()

    #RBF
    def rbf_kernel_and_grad(X, sigma):
        N, D = X.size()
        X_i = X.unsqueeze(1)
        X_j = X.unsqueeze(0)
        dist_sq = torch.sum((X_i - X_j)**2, dim=2)
       
        K = torch.exp(-dist_sq / (2.0 * sigma**2))
        X_diff = X_j - X_i
        grad_K_i = K.unsqueeze(2) * X_diff / (sigma**2)
        sum_grad_K = torch.sum(grad_K_i, dim=1)
       
        return K.detach(), sum_grad_K.detach()

#trainer
class SVGDRegressionTrainer:
    def __init__(self, n_particles=10):
        self.n_particles = n_particles
        self.models = [RegressionNet() for _ in range(n_particles)]
        self.svgd = SVGD()

    def get_theta(self):
        #all model weights into an (N, D) matrix.
        all_params = []
        for model in self.models:
            p_vector = torch.cat([p.data.view(-1) for p in model.parameters()])
            all_params.append(p_vector)
        return torch.stack(all_params)

    def update_models(self, theta_new):
        #Puts updated weights back into the models.
        for i, model in enumerate(self.models):
            pointer = 0
            for p in model.parameters():
                num = p.numel()
                p.data.copy_(theta_new[i, pointer:pointer + num].view(p.shape))
                pointer += num

    def compute_loss_gradients(self, X, y):
        #Log-Likelihood Gradient
        gradients = []
        total_loss = 0
        for model in self.models:
            model.zero_grad()
            pred = model(X)
            loss = nn.MSELoss()(pred, y)
            total_loss += loss.item()
            
            (-loss).backward()
           
            g_vector = torch.cat([p.grad.view(-1) for p in model.parameters()])
            gradients.append(g_vector)
       
        avg_loss = total_loss / self.n_particles
        return torch.stack(gradients), avg_loss

    def train_step(self, X, y, lr=0.001):#learning rate:0.001
        # current weights
        theta = self.get_theta()
       
        #Get loss Gradient and likehood
        ln_p_grad, current_loss = self.compute_loss_gradients(X, y)
       
        #Repulsive Force (Kernel logic)
        sigma = self.svgd.median_heuristic(theta)
        K, sum_grad_K = self.svgd.rbf_kernel_and_grad(theta, sigma)
       
       
        phi = (torch.matmul(K, ln_p_grad) + sum_grad_K) / self.n_particles
       
        #push back to models
        theta_new = theta + lr * phi
        self.update_models(theta_new)
       
        return current_loss # Now 'current_loss' is defined and safe to return
   
#Data load
if __name__ == "__main__":
    
    file_path = r"C:\Users\emma\Downloads\2.1_2.2_Final_Dataset_City_Level (1).xlsx"
    sheet_name = "2.1_City_Daily_Missing"

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['Date'] = pd.to_datetime(df['Date'])
   
    
    df['Ct_Value'] = pd.to_numeric(df['Ct_Value'].replace('NA', np.nan), errors='coerce')
    df['NewCases'] = pd.to_numeric(df['NewCases'], errors='coerce')
    df = df.dropna(subset=['NewCases', 'Ct_Value'])

    
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

#visualiztion
sorted_indices = torch.argsort(X.squeeze())
X_sorted = X[sorted_indices]
y_sorted = y[sorted_indices]

#!!predictions with each model
all_predictions = []
for model in trainer.models:
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        predictions = model(X_sorted).squeeze().numpy()
    all_predictions.append(predictions)

all_predictions = np.array(all_predictions)

#mean and standard deviation of predictions
mean_predictions = np.mean(all_predictions, axis=0)
std_predictions = np.std(all_predictions, axis=0)

# plot
plt.figure(figsize=(12, 7))

# mean prediction
plt.plot(X_sorted.numpy(), mean_predictions, color='blue', label='Mean Prediction')

# uncertainty
plt.fill_between(X_sorted.numpy().squeeze(),
                 mean_predictions - std_predictions,
                 mean_predictions + std_predictions,
                 color='skyblue', alpha=0.4, label='Uncertainty (1 Std Dev)')

# real data points
plt.scatter(X_sorted.numpy(), y_sorted.numpy(), color='red', s=10, label='Actual Data', alpha=0.6)

plt.title('SVGD Bayesian Regression: Mean Prediction with Uncertainty')
plt.xlabel('Scaled NewCases')
plt.ylabel('Scaled Ct_Value')
plt.legend()
plt.grid(True)
plt.show()