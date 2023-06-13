import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
print(n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#1 model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1) 
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegression(n_features)

#2 loss and optimizer
n_iter = 1000
learning_rate = .001
criterion = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3 training loops
for epoch in range(n_iter):

    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    # loss = criterion(torch.clamp(y_train+1e-4,0,1),\
    #                  torch.clamp(y_pred+1e-4,0,1))

    # backward pass
    loss.backward() # calculate dL/dw

    # update
    optimizer.step()

    # zero grads
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum()/ float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
# plt.plot(X_numpy, y_numpy, 'ro')
# plt.plot(X_numpy, predicted, 'b')
# plt.show()

# print(f'Prediction after training: f(5) = {model(X_test).item()}')  