import torch
from torch.utils import data as data
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from dbn import DBN
from rbm import RBM

BATCH_SIZE = 64
EPOCHS = 5

def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


data_transform = transforms.Compose([
    # transforms.Resize(84),
    # transforms.CenterCrop(84),
    transforms.ToTensor(),  # 从图片读取像素并转化成0-1数字
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 图片数据归一化
])

DATA_FOLDER = './mnist'
train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, download=False, transform=data_transform)

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False, download=False, transform=data_transform)

test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)

#
# data_iter = iter(train_loader)
# images, labels = data_iter.next()
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(64)))

net = DBN()

print(net)

print(len(net.rbm_layers))

for i in range(len(net.rbm_layers)):
    print("-------------No. {} layer's weights-------------".format(i+1))
    print(net.rbm_layers[i].weights)
    print(len(net.rbm_layers[i].weights))
    print("-------------No. {} layer's visible_bias-------------".format(i+1))
    print(net.rbm_layers[i].visible_bias)
    print(len(net.rbm_layers[i].visible_bias))
    print("-------------No. {} layer's hidden_bias-------------".format(i+1))
    print(net.rbm_layers[i].hidden_bias)
    print(len(net.rbm_layers[i].hidden_bias))
    print("-------------No. {} layer's Learning rate-------------".format(i+1))
    print(net.rbm_layers[i].learning_rate)


train_features, train_labels = net.train_static(train_loader, train_dataset)
print(train_features)
print(train_labels)

test_features, test_labels = net.Testing(test_loader, test_dataset)
print(test_features)
print(test_labels)

