import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # first column is class
        self.n_samples = xy.shape[0]
    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # get length
        return self.n_samples

datset = WineDataset()

first_data = datset[0]
features, labels = first_data

dataloader = DataLoader(dataset=datset, batch_size=15, shuffle=True, num_workers=2)

dataiter = iter(dataloader)
data = dataiter.next()

features, labels = data #for batch

# training loop

num_epochs = 2
total_samples = len(datset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        #forward, backward, update, etc
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step: {i+1}/{n_iterations}')
            print(inputs.shape)
