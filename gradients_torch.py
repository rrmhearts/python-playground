import torch

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([3,6,9,12], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

def loss(y, y_pred):
    return torch.dot(y_pred-y, y_pred-y)

# L = (w*x-y)**2 (avg)
def gradient(x, y, y_pred):
    return 2*torch.dot(x, y_pred-y)

print(f"Prediction before training: f(5) = {forward(5)}")

n_iter = 20
learning_rate = .01

for epoch in range(n_iter):
    y_pred = forward(X)

    l = loss(Y, y_pred)

    # dw = gradient(X, Y, y_pred)
    l.backward() # calculate dL/dw

    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero grads
    w.grad.zero_()

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5)}')