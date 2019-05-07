import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class CIFAREncoderV3(nn.Module):
    def __init__(self, hidden_layer_size):
        super(CIFAREncoderV3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.scale1 = ScaleLayer()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.scale2 = ScaleLayer()
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.scale3 = ScaleLayer()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.scale4 = ScaleLayer()
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.scale5 = ScaleLayer()
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.scale6 = ScaleLayer()
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.scale7 = ScaleLayer()
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.scale8 = ScaleLayer()
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        self.scale9 = ScaleLayer()
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(128)
        self.scale10 = ScaleLayer()
        self.conv11 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(128)
        self.scale11 = ScaleLayer()
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(128)
        self.scale12 = ScaleLayer()
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.scale13 = ScaleLayer()
        self.fc1 = nn.Linear(128*1*1, hidden_layer_size)
        
    def forward(self, x):
        x = (x - 0.5) * 2
        x = F.relu(self.scale1(self.bn1(self.conv1(x))))
        x = F.relu(self.scale2(self.bn2(self.conv2(x))))
        x = F.max_pool2d(x,2)
        x = F.relu(self.scale3(self.bn3(self.conv3(x))))
        x = F.max_pool2d(x,2)
        x = F.relu(self.scale4(self.bn4(self.conv4(x))))
        x = F.max_pool2d(x,2)
        x = F.relu(self.scale5(self.bn5(self.conv5(x))))
        x = F.relu(self.scale6(self.bn6(self.conv6(x))))
        x = F.relu(self.scale7(self.bn7(self.conv7(x))))
        x = F.max_pool2d(x,2)
        x = F.relu(self.scale8(self.bn8(self.conv8(x))))
        x = F.max_pool2d(x,2)
        x = F.relu(self.scale9(self.bn9(self.conv9(x))))
        x = F.max_pool2d(x,2)
        x = F.relu(self.scale10(self.bn10(self.conv10(x))))
        x = F.relu(self.scale11(self.bn11(self.conv11(x))))
        x = F.relu(self.scale12(self.bn12(self.conv12(x))))
        x = F.max_pool2d(x,2)
        x = F.relu(self.scale13(self.bn13(self.conv13(x))))
        x = F.max_pool2d(x,2)
        x = x.view(-1,128*1*1)
        x = F.relu(self.fc1(x))
        return x
    
class CIFARClassifierV3(nn.Module):
    def __init__(self, encoder, hidden_layer_size):
        super(CIFARClassifierV3, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(hidden_layer_size, 10)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class CIFARDecoderV3(nn.Module):
    def __init__(self, hidden_layer_size):
        super(CIFARDecoderV3, self).__init__()
        self.fc1 = nn.Linear(hidden_layer_size, 256*8*8)
        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv7 = nn.Conv2d(128, 3, 3, 1, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = torch.sigmoid(self.conv7(x))
        return x

class CIFARAutoEncoderV3(nn.Module):
    def __init__(self, hidden_layer_size=128):
        super(CIFARAutoEncoderV3, self).__init__()
        self.encoder = CIFAREncoderV3(hidden_layer_size)
        self.decoder = CIFARDecoderV3(hidden_layer_size)
        self.classifier = CIFARClassifierV3(self.encoder, hidden_layer_size)
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
    
    train_load, test_load = get_cifar_loaders(download=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = CIFARAutoEncoderV3(256).to(device)


    opt_enc = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=0.001)

    epoch_enc = 0
    epoch_dec = 0


    for _ in range(20):
        train_encoder(model, device, train_load, opt_enc, epoch_enc)
        test_encoder(model, device, test_load)
        epoch_enc += 1

    for _ in range(99):
        train_decoder(model, device, train_load, opt_dec, epoch_dec)
        test_decoder(model, device, test_load)
        epoch_dec += 1

    show_true_and_recreated_imgs(model, train_load, device)
    
    show_transition(model, train_load, device)