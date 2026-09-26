# ------------------------------------------------
# Sources
# ------------------------------------------------
# PINN:             https://vizuara.medium.com/an-introduction-to-physics-informed-neural-networks-pinns-teach-your-neural-network-to-respect-af484ac650fc
# Sequential:       https://medium.com/we-talk-data/pytorchs-sequential-3974f27c714e
# Custom training:  https://medium.com/biased-algorithms/creating-a-training-loop-for-pytorch-models-96e260e70766

# ------------------------------------------------
# Includes
# ------------------------------------------------
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("c:/Users/ext-pourb/Documents/own/Tudomány/Egyetem/Doktori/5_3_NeuralSS_Static/0_ss_mdl/ss_mdl.py"))))
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import signal
from sys_class import *
from sys_nn_class import *
from sys_ss_class import *
from CustomSysNNModel import *
from schroeder import *
from read_meas import *
from early_stopping_class import *
import pandas as pd

import gpflow
from trieste.space import Box
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import ExpectedImprovement
from trieste.bayesian_optimizer import BayesianOptimizer

# ------------------------------------------------
# Configuration
# ------------------------------------------------
display_on = 1
file = '/home/pourb/Documents/Doktori/5_3_NeuralSS_Static/Meas2csv/3000Nm_100degs_1Nm_V1_FlowAdded.csv'
train_amount = 1

# ------------------------------------------------
# Private functions
# ------------------------------------------------
def noise():
    return np.random.normal(0,0.01)

def loss_mse(y, y_hat):
    return torch.mean(torch.square(y_hat - y))

# ------------------------------------------------
# Time Series
# ------------------------------------------------
Ts = get_meas_sampletime(file)
N = int(get_meas_length(file)*train_amount)

t = np.linspace(Ts,N*Ts,N)
u = get_meas_in(file)[0:N]
y = get_meas_out(file)[0:N]
s = get_states(file)[0:N]
su = get_states_in(file)[0:N]

# Create dataset
dataset = TensorDataset(torch.from_numpy(su[0:N-1,:]),torch.from_numpy(s[1:N,:]))

# ------------------------------------------------
# Neural Network Model
# ------------------------------------------------
##N_i, N_f, N_h1, N_h2, N_o = 9, 9, 21, 21, 1
##N_i, N_f, N_h1, N_h2, N_o = 2, 2, 5, 5, 1
# ------------------------------------------------
# Custom Train
# ------------------------------------------------
# Instantiate a loss function.
##opt_nn_mdl = torch.optim.Adam(sys_nn_mdl.parameters(), lr=0.0001)

# 1. Define Search Space
#  Say we are tuning 2 hyperparameters, each ranging from -2 to 2
search_space = Box(
    lower = [1, 1, -6, -10, 1],   # N_h1, N_h2, lr, weight_decay(L2 norm), batch size
    upper = [1000, 1000, -2, -3, N])

# 2. Define the Observer (Our black-box model)
#  In practice, this function takes your hyperparameters, trains your ML model,
#  and returns the validation loss.
def custom_train(x):
    N_h1 = int(x[0])
    N_h2 = int(x[1])
    lr = 10**x[2].numpy()
    wd = 10**x[3].numpy()
    bs = int(x[4].numpy())

    print(N_h1, N_h2, lr, wd, bs)

    # Create batches
    loader = DataLoader(dataset, batch_size=bs)

    # Define system
    sys_pinss_mdl = sys_pinss(s[0],1,N_h1,N_h2)
    C = np.reshape(np.array([0.0,0.0,0.0,1.0]),(1,4))
    y_pinss_mdl_inference = np.zeros((N,1))
    opt_pinss_mdl = torch.optim.SGD(sys_pinss_mdl.parameters(),lr=lr,weight_decay=wd)
    sys_pinss_mdl.train()

    # Define early stopping
    early_stopping = EarlyStopping(patience=3, mode="min")
    final_loss = 1000000000

    for epoch in range(10):
        print(f"Epoch: {epoch}")

        # Train
        for xb, yb in loader:

            # Clear accumulated gradients
            opt_pinss_mdl.zero_grad()

            # Run Nerual Network
            out_pinss_mdl = sys_pinss_mdl(xb)

            # Calculate losses
            loss_pinss = loss_mse(yb, out_pinss_mdl)

            # Compute gradients
            loss_pinss.backward()
            
            # Update model parameters
            opt_pinss_mdl.step()

        # Validation
        sys_pinss_mdl.setstate(s[0])
        for i in range(N):
            y_pinss_mdl_inference[i] = np.matmul(C,np.transpose(sys_pinss_mdl.state))
            out_pinss_mdl = sys_pinss_mdl(torch.tensor(np.concatenate((sys_pinss_mdl.state,np.reshape(u[i],(1,1))),axis=1)))
        final_loss = loss_mse(y_pinss_mdl_inference, torch.tensor(np.reshape(y,(len(y),1))))

        if np.isnan(final_loss):
            final_loss = 1000000000
        early_stopping(final_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print(final_loss)
    print("\n")
    return final_loss

def observer(query_points):
    losses = np.zeros((query_points.shape[0],1))
    for i in range(query_points.shape[0]):
        losses[i] = custom_train(query_points[i])
    return Dataset(query_points, losses)

# 3. Generate initial random data
num_initial_points = 5
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# 4. Define the Probabilistic Model (Gaussian Process)
##kernel = gpflow.kernels.Matern52()
kernel = gpflow.kernels.SquaredExponential()
gpr = gpflow.models.GPR(
    data=(initial_data.query_points, initial_data.observations),
    kernel=kernel,
    mean_function=None)
gp_model = GaussianProcessRegression(gpr)

# 5. Choose the Acquisition Rule & Strategy
acquisition_rule = EfficientGlobalOptimization(
    builder=ExpectedImprovement())

# 6. Run the Optimizer
bo = BayesianOptimizer(observer, search_space)
num_steps = 10
result = bo.optimize(num_steps, initial_data, gp_model, acquisition_rule)

# 7. Extract the best performing hyperparameters
dataset = result.try_get_final_dataset()
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

best_idx = np.argmin(observations)
best_params = query_points[best_idx]
best_value = observations[best_idx]

print("Best parameters: ", best_params)
print("Best objective: ", best_value)

# Displays timeseries
if display_on:    
    plt.plot(t,u)
    plt.plot(t,y)
    plt.plot(t,y_pinss_mdl_inference)
    plt.legend(["u","y","y_nss_mdl_inference"])
    plt.title('Timeseries')
    plt.xlabel('time [s]')
    plt.show()

# ------------------------------------------------
# Show Statistics
# ------------------------------------------------
# Numpy to dataframe
names_df = ['N_h1', 'N_h2', 'lr', 'wd', 'bs', 'loss']
results_data = np.concatenate((best_params, best_value), axis=0)
results_data = np.reshape(results_data, (1,results_data.shape[0]))
results_data = np.concatenate((results_data, np.concatenate((query_points, observations), axis=1)), axis=0)
results_df = pd.DataFrame(results_data, columns=names_df)

# ------------------------------------------------
# Save Statistics
# ------------------------------------------------
results_df.to_csv("/home/pourb/Documents/Doktori/5_3_NeuralSS_Static/0_2_pinss_mdl/Results/3000Nm_100degs_1Nm_V1_FlowAdded_tanh_SGD.csv", float_format="%.6f")
