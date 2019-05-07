import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class CIFAREncoderUNet(nn.Module):
    def __init__(self, hidden_layer_size):
        super(CIFAREncoderUNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        
        self.fc1 = nn.Linear(128*8*8, hidden_layer_size)
        
        self.tll1 = nn.Linear(32*32*32, 32)
        self.tll2 = nn.Linear(64*16*16, 32)
        
    def forward(self, x):
        x = (x - 0.5) * 2
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        out1 = x.view(-1, 32*32*32)
        out1 = F.relu(self.tll1(out1))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        out2 = x.view(-1, 64*16*16)
        out2 = F.relu(self.tll2(out2))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1,128*8*8)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, out1, out2), 1)
        return x
    
class CIFARClassifierUNet(nn.Module):
    def __init__(self, encoder, hidden_layer_size):
        super(CIFARClassifierUNet, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(hidden_layer_size+64, 10)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class CIFARDecoderUNet(nn.Module):
    def __init__(self, hidden_layer_size):
        super(CIFARDecoderUNet, self).__init__()
        self.fc1 = nn.Linear(hidden_layer_size, 128*32*32)
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
#         self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
#         self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv7 = nn.Conv2d(128, 3, 3, 1, 1)
    
        self.tll1 = nn.Linear(32, 32*32*128)
        self.tll2 = nn.Linear(32, 32*32*128)
        
        self.hidden_layer_size = hidden_layer_size
        
    def forward(self, x):
        h = self.hidden_layer_size
        out1 = x[:, h:(h + 32)]
        out1 = F.relu(self.tll1(out1))
        out2 = x[:, (h + 32):(h + 64)]
        out2 = F.relu(self.tll2(out2))
        x = x[:, 0:h]
        x = F.relu(self.fc1(x))
        x = x.view(-1, 128, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x + out1.view(-1, 128, 32, 32)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x + out2.view(-1, 128, 32, 32)
#         x = F.relu(self.conv5(x))
#         x = F.relu(self.conv6(x))
        x = torch.sigmoid(self.conv7(x))
        return x

class CIFARAutoEncoderUNet(nn.Module):
    def __init__(self, hidden_layer_size=128):
        super(CIFARAutoEncoderUNet, self).__init__()
        self.encoder = CIFAREncoderUNet(hidden_layer_size)
        self.decoder = CIFARDecoderUNet(hidden_layer_size)
        self.classifier = CIFARClassifierUNet(self.encoder, hidden_layer_size)
        self.hidden_layer_size = hidden_layer_size
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def train_encoder(model, device, train_loader, optimizer, epoch):
    """
    Trains the encoder via the classifier in the model
    
    params:
        model: torch model to train
        device: device on which to train
        train_loader: loader for the training data
        optimizer: optimizer to use in training
        epoch: integer of current epoch (only used for printing)
    """
    
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.classifier(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test_encoder(model, device, test_loader):
    """
    Tests the encoder via classifier for accuracy
    
    params:
        model: torch model to test
        device: device on which to test
        test_loader: data on which to test
    """
    
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.classifier(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def train_decoder(model, device, train_loader, optimizer, epoch):
    """
    Trains the decoder, holding weights in the encoder constant.
    
    params:
        model: pytorch model whose decoder is to be trained
        device: device on which to train
        train_loader: loader for the training data set
        optimizer: optimizer to use for training decoder
        epoch: integer of current epoch (only for printing)
    
    """
    
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        encoded = model.encoder(data).detach()
        output = model.decoder(encoded)
        loss = F.binary_cross_entropy(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_decoder(model, device, test_loader):
    """
    Gives an indication of how the decoder is doing by showing a sample of
    data and their recreations.
    
    params:
        model: pytorch model which to test
        device: device on which to test
        test_loader: loader for the data on which to test
    """
    
    
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
    
def train(model, device, train_loader, optimizer, epoch):
    """
    Trains the entire model together.
    
    params:
        model: pytorch model to train
        device: device on which to train
        train_loader: loader for data on which to train
        optimizer: optimizer for parameter update rule
        epoch: integer of current epoch (used only for printing)
    
    """
    
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
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
    """
    Gives an indication of how the entire model is doing by 
    showing a sample of data and their recreations.
    
    params:
        model: pytorch model which to test
        device: device on which to test
        test_loader: loader for the data on which to test
    
    """
    
    
    
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
    
    
def get_cifar_loaders(location='data', use_cuda=False, download=True, train_batch_size=20, test_batch_size=20):
    """
    Returns loaders for the CIFAR-10 dataset.
    """
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
    """
    Shows a row of original images and a row of recreated images
    
    params:
        model: pytorch model to use in recreations
        loader: loader from which to sample data to recreate
        device: device on which it all lives
        n: number of images to recreate
    """
    
    
    
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
    """
    Samples two images from the data, and shows recreation of the transition
    in encoded space between the two images.
    
    params:
        model: pytorch model to use in recreation
        loader: data from which to sample
        device: device on which everything lives
        n: number of intermediate steps in transition
    
    """
    
    
    
    encoded_vecs = torch.zeros([n, model.hidden_layer_size+64])

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
            encoded_vecs[i] = ((n-i-1) / (n-1)) * encoded_vecs[0] + (i/(n-1)) * encoded_vecs[n-1]

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
    model = CIFARAutoEncoderUNet(80).to(device)
    train_load, test_load = get_cifar_loaders(download=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epoch = 0
    
    for _ in range(80):
        train(model, device, train_load, opt, epoch)
        test(model, device, test_load)
        epoch_enc += 1
        
    
    show_true_and_recreated_imgs(model, train_load, device)
    
    show_transition(model, train_load, device)