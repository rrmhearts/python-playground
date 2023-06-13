import urllib.request as req
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://pytorch.tips/coffee'
fpath = 'coffee.jpg'
req.urlretrieve(url, fpath)

img = Image.open(fpath)
plt.imshow(img)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224, 0.225)),
])

img_tensor = transform(img)
print(type(img_tensor), img_tensor.shape)

batch = img_tensor.unsqueeze(0)
print(batch.shape)

model = models.alexnet(pretrained=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

y = model(batch.to(device))

y_max, index = torch.max(y,1)
print(index, y_max)

url_classes = 'https://pytorch.tips/imagenet-labels'
fpath = 'imagenet_labels.txt'
req.urlretrieve(url_classes, fpath)

with open(fpath, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

indexNum = index[0].item()
print("index", index, indexNum)
print(classes[index[0]], classes[index])

prob = torch.nn.functional.softmax(y, dim=1)[0] * 100
print(classes[index], prob[index].item())

_, indices = torch.sort(y, descending=True)

for i in indices[0][:5]:
    print(classes[i], prob[i].item())
    