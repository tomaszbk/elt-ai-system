import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


laptops_data_df = pd.read_sql("select * from laptop_dim", "postgresql://airflow:airflow@elt-ai-system-postgres-1:5432/laptops")
 

X = laptops_data_df.drop(['price'], axis=1)
y = laptops_data_df['price']

# train-test split for model evaluation
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
 
# Standardizing data
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
 
# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
 
# Define the model
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)
 
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)
 

num_of_epochs=1000
for i in range(num_of_epochs):
  # give the input data to the architecure
  y_train_prediction=model(X_train)  # model initilizing
  loss_value = loss_fn(y_train_prediction.squeeze(),y_train)   # find the loss function:
  optimizer.zero_grad() # make gradients zero for every iteration so next iteration it will be clear
  loss_value.backward()  # back propagation
  optimizer.step()  # update weights in NN

  # print the loss in training part:
  if i % 10 == 0:
    print(f'[epoch:{i}]: The loss value for training part={loss_value}')


with torch.no_grad():
  model.eval()   # make model in evaluation stage
  y_test_prediction=model(X_test)
  test_loss = loss_fn(y_test_prediction.squeeze(),y_test)
  print(f'Test loss value : {test_loss.item():.4f}')