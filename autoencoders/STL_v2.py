import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class STLEncoder(nn.Module):
    def __init__(self, hidden_layer_size, tll_size):
        super(STLEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv11 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv12 = nn.Conv2d(128, 128, 3, 1, 1)
        
        self.fc1 = nn.Linear(128*12*12, hidden_layer_size)
        
        self.tll_size = tll_size
        
        self.tll1 = nn.Linear(128*48*48, self.tll_size)
        self.tll2 = nn.Linear(128*24*24, self.tll_size)
        
    def forward(self, x):
        x = (x - 0.5) * 2
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        out1 = x.view(-1, 128*48*48)
        out1 = F.relu(self.tll1(out1))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        out2 = x.view(-1, 128*24*24)
        out2 = F.relu(self.tll2(out2))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = x.view(-1,128*12*12)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, out1, out2), 1)
        return x
    
class STLDecoder(nn.Module):
    def __init__(self, hidden_layer_size, tll_size):
        super(STLDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_layer_size, 128*12*12)
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv11 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv12 = nn.Conv2d(128, 3, 3, 1, 1)
    
        self.tll1 = nn.Linear(tll_size, 48*48*128)
        self.tll2 = nn.Linear(tll_size, 24*24*128)
        
        self.hidden_layer_size = hidden_layer_size
        self.tll_size = tll_size
        
    def forward(self, x):
        h = self.hidden_layer_size
        t = self.tll_size
        out1 = x[:, h:(h + t)]
        out1 = F.relu(self.tll1(out1))
        out2 = x[:, (h + t):(h + 2*t)]
        out2 = F.relu(self.tll2(out2))
        x = x[:, 0:h]
        x = F.relu(self.fc1(x))
        x = x.view(-1, 128, 12, 12)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, size=(24, 24))
        x = x + out2.view(-1, 128, 24, 24)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.interpolate(x, size=(48, 48))
        x = x + out1.view(-1, 128, 48, 48)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.interpolate(x, size=(96, 96))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = torch.sigmoid(self.conv12(x))
        return x

class STLAutoEncoder(nn.Module):
    def __init__(self, hidden_layer_size=128, tll_size=32):
        super(STLAutoEncoder, self).__init__()
        self.encoder = STLEncoder(hidden_layer_size, tll_size)
        self.decoder = STLDecoder(hidden_layer_size, tll_size)
        self.hidden_layer_size = hidden_layer_size
        self.tll_size = tll_size
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    n = 10
    
    x_test = []
    decoded_imgs = []

    denormalize = lambda x : np.transpose(x, (1, 2, 0))

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            x_test += [data.cpu().numpy()[0,:,:,:]]
            decoded_imgs += [output.cpu().numpy()[0,:,:,:]]
            if len(x_test) == n:
                break

    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(denormalize(x_test[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(denormalize(decoded_imgs[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
def get_stl_loaders(location='data', use_cuda=False, download=True, train_batch_size=20, test_batch_size=20):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose(
        [transforms.ToTensor()])
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_loader = torch.utils.data.DataLoader(
        datasets.STL10('data', split='train+unlabeled', download=download, transform=transform),
        batch_size=train_batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.STL10('data', split='test', transform=transform),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader

def denormalize(x):
    return np.transpose(x, (1, 2, 0))

def show_true_and_recreated_imgs(model, loader, device, n=10):
    model.eval()
    x_test = []
    decoded_imgs = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            output = model(data)
            x_test += [data.cpu().numpy()[0,:,:,:]]
            decoded_imgs += [output.cpu().numpy()[0,:,:,:]]
            if len(x_test) == n:
                break
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(denormalize(x_test[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(denormalize(decoded_imgs[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def show_transition(model, loader, device, n=10):
    encoded_vecs = torch.zeros([n, model.hidden_layer_size + 2*model.tll_size])

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            encoded = model.encoder(data)
            break

        first_img = data[0].cpu().numpy()
        second_img = data[1].cpu().numpy()

        encoded_vecs[0] = encoded[0]
        encoded_vecs[n-1] = encoded[1]
        for i in range(1, n-1):
            encoded_vecs[i] = ((n-i) / n) * encoded_vecs[0] + (i/n) * encoded_vecs[n-1]

        encoded_vecs = encoded_vecs.to(device)
        imgs = model.decoder(encoded_vecs)
        imgs = imgs.cpu().numpy()

    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        # display interpolation
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(denormalize(imgs[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    ax = plt.subplot(2, n, n+1)
    plt.imshow(denormalize(first_img))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, 2*n)
    plt.imshow(denormalize(second_img))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()