import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
print(X_numpy.shape, y_numpy.shape)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
print(X.size(), y.size())
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

#1 model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(n_features, 1)

# print(f"Prediction before training: f(5) = {model(X_test).item()}")

#2 loss and optimizer
n_iter = 100
learning_rate = .01

criterion = nn.MSELoss() 
# optimizer = torch.optim.SGD([w], lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3 training loops
for epoch in range(n_iter):

    # forward pass and loss
    y_pred = model(X)

    loss = criterion(y, y_pred)

    # backward pass
    loss.backward() # calculate dL/dw

    # update
    optimizer.step()

    # zero grads
    optimizer.zero_grad()
    [w, b] = model.parameters()
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: loss = {loss.item():.4f}')

predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

# print(f'Prediction after training: f(5) = {model(X_test).item()}') 