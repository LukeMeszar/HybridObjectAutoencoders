import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class CIFAREncoderV1(nn.Module):
    def __init__(self, hidden_layer_size):
        super(CIFAREncoderV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, 5, 1, 2)
        self.conv2 = nn.Conv2d(30, 72, 5, 1, 2)
        self.fc1 = nn.Linear(72*8*8, hidden_layer_size)
        
    def forward(self, x):
        x = (x - 0.5) * 2
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1,72*8*8)
        x = F.relu(self.fc1(x))
        return x
    
class CIFARDecoderV1(nn.Module):
    def __init__(self, hidden_layer_size):
        super(CIFARDecoderV1, self).__init__()
        self.fc1 = nn.Linear(hidden_layer_size, 72*8*8)
        self.conv1 = nn.Conv2d(72, 60, 5, 1, 2)
        self.conv2 = nn.Conv2d(60, 60, 5, 1, 2)
        self.conv3 = nn.Conv2d(60, 30, 5, 1, 2)
        self.conv4 = nn.Conv2d(30, 3, 5, 1, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 72, 8, 8)
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

class CIFARAutoEncoderV1(nn.Module):
    def __init__(self, hidden_layer_size=128):
        super(CIFARAutoEncoderV1, self).__init__()
        self.encoder = CIFAREncoderV1(hidden_layer_size)
        self.decoder = CIFARDecoderV1(hidden_layer_size)
        self.hidden_layer_size = hidden_layer_size
    
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
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, data).item() # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Total loss: {:.4f}\n'.format(test_loss))
    
def get_cifar_loaders(location='data', use_cuda=False, download=True, train_batch_size=20, test_batch_size=20):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose(
        [transforms.ToTensor()])
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=download, transform=transform),
        batch_size=train_batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform),
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
    encoded_vecs = torch.zeros([n, model.hidden_layer_size])

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
    
if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CIFARAutoEncoderV1(80).to(device)
    train_load, test_load = get_cifar_loaders(download=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epoch = 0
   
    
    for _ in range(80):
        train(model, device, train_load, opt, epoch)
        test(model, device, test_load)
        epoch_enc += 1
        
    
    show_true_and_recreated_imgs(model, train_load, device)
    
    show_transition(model, train_load, device)