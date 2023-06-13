import numpy as np

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([3,6,9,12], dtype=np.float32)

w = 0.0

def forward(x):
    return w * x

def loss(y, y_pred):
    return np.dot(y_pred-y, y_pred-y)

# L = (w*x-y)**2 (avg)
def gradient(x, y, y_pred):
    return 2*np.dot(x, y_pred-y)

print(f"Prediction before training: f(5) = {forward(5)}")

n_iter = 30
learning_rate = .01

for epoch in range(n_iter):
    y_pred = forward(X)

    l = loss(Y, y_pred)

    dw = gradient(X, Y, y_pred)

    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5)}')