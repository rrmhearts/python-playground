import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[3],[6],[9],[12]], dtype=torch.float32)

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
# def forward(x):
#     return w * x

# def loss(y, y_pred):
#     return torch.dot(y_pred-y, y_pred-y)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
# print(X.shape)
model = nn.Linear(n_features, n_features) #req 2D input
print(f"Prediction before training: f(5) = {model(X_test).item()}")

n_iter = 1000
learning_rate = .1

loss = nn.MSELoss() 
# optimizer = torch.optim.SGD([w], lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(n_iter):
    y_pred = model(X)

    l = loss(Y, y_pred)

    l.backward() # calculate dL/dw

    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()

    # zero grads
    # w.grad.zero_()
    optimizer.zero_grad()
    [w, b] = model.parameters()
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item()}') 