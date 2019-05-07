"""
This model was adopted from the model in https://github.com/atinghosh/VAE-pytorch/blob/master/VAE_CNN_BCEloss.py
"""
import os
import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from VAE_mnist_v1 import VAE_mnist
from VAE_CIFAR_v1 import VAE_CIFAR_v1
from VAE_CIFAR_v2 import VAE_CIFAR_v2
"""
Setup
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = True
SEED = 1
BATCH_SIZE = 16
LOG_INTERVAL = 100
EPOCHS = 25
no_of_sample = 10
ZDIMS = 20
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

def load_data():
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
    if DATASET == 0:
        print("Load Data: MNIST")
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=True, download=True,transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        return train_loader, test_loader
    elif DATASET == 1:
        print("Load Data: CIFAR")
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./CIFAR10', train=True, download=True,transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./CIFAR10', train=False, transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        return train_loader, test_loader

    elif DATASET == 2:
        print("Load Data: CIFAR10+CARS_PROCESSED")
        #train_url = 'https://grantbaker.keybase.pub/data/background_filtered/train_imagefolder.zip'
        #test_url = 'https://grantbaker.keybase.pub/data/background_filtered/test_imagefolder.zip'
        #train_filename = wget.download(train_url, bar=bar_thermometer)
        #test_filename = wget.download(test_url, bar=bar_thermometer)+
        processed_CIFAR10_data_train = datasets.ImageFolder(root='train_imagefolder/', transform=transforms.ToTensor())
        processed_CIFAR10_data_test = datasets.ImageFolder(root='test_imagefolder/', transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(processed_CIFAR10_data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(processed_CIFAR10_data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        return train_loader, test_loader
    else:
        print("Invalid dataset, needs to be 0,1,2")
        sys.exit(1)

def load_model_and_optimizer():
    if DATASET == 0:
        print("Create model for: MNIST")
        model = VAE_mnist(ZDIMS, BATCH_SIZE, no_of_sample)
    elif DATASET == 1:
        print("Create model for: CIFAR10")
        model = VAE_CIFAR_v1(ZDIMS, BATCH_SIZE, no_of_sample)
    elif DATASET == 2:
        print("Create model for: CIFAR10+CARS_PROCESSED")
        model = VAE_CIFAR_v2(ZDIMS, BATCH_SIZE, no_of_sample)
    if CUDA:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer

def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if CUDA:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.9f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(epoch, model, test_loader):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if CUDA:
            data = data.cuda()

        # we're only going to infer, so no autograd at all required which means
        with torch.no_grad():
            data = Variable(data)
        recon_batch, mu, logvar = model(data)
        test_loss += model.loss_function(recon_batch, data, mu, logvar).item()
        if i == 0:
            n = min(data.size(0), 8)
            # for the first 128 batch of the epoch, show the first 8 input digits
            # with right below them the reconstructed output digits based on dataset
            if DATASET == 0:
                comparison = torch.cat([data[:n],
                                        recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.data.cpu(),
                           './mnist/reconstruction_' + str(epoch) + '.png', nrow=n)
            elif DATASET == 1:
                comparison = torch.cat([data[:n],
                                        recon_batch.view(BATCH_SIZE, 3, 32, 32)[:n]])
                save_image(comparison.data.cpu(),
                           './CIFAR10/reconstruction_' + str(epoch) + '.png', nrow=n)
            elif DATASET == 2:
                comparison = torch.cat([data[:n],
                                        recon_batch.view(BATCH_SIZE, 3, 50, 50)[:n]])
                save_image(comparison.data.cpu(),
                           './cifar10_processed/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def train_and_test(model, optimizer, train_loader, test_loader):
    for epoch in range(1, EPOCHS + 1):
        train(epoch, model, optimizer, train_loader)
        test(epoch, model, test_loader)


        #64 sets of random ZDIMS-float vectors, i.e. 64 locations of  data in latent space for
        #visualizing progress

        sample = Variable(torch.randn(64, ZDIMS))
        if CUDA:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()

        if DATASET == 0:
            save_image(sample.data.view(64, 1, 28, 28),'./mnist/reconstruction' + str(epoch) + '.png')
        elif DATASET == 1:
            save_image(sample.data.view(64, 3, 32, 32),'./CIFAR10/reconstruction' + str(epoch) + '.png')
        elif DATASET == 2:
            save_image(sample.data.view(64, 3, 100, 100),'./cifar10_processed/reconstruction' + str(epoch) + '.png')

def interpolate_images(model):
    num_interpolation_points = 16 #should be a multiple of 8 to make formatting nice
    sample = Variable(torch.randn(2, ZDIMS))
    first_point = sample[0]
    last_point = sample[1]
    interpolation_points_list = []
    #create interpolation points liunearly
    for i in np.linspace(0,1,num_interpolation_points):
        new_interpolation_point = (1-i)*first_point+i*last_point
        interpolation_points_list.append(new_interpolation_point)

    interpolation_sample = Variable(torch.stack(interpolation_points_list))
    if CUDA:
        interpolation_sample = interpolation_sample.cuda()
    interpolation_sample = model.decode(interpolation_sample).cpu()
    if DATASET == 0:
        save_image(interpolation_sample.data.view(num_interpolation_points, 1, 28, 28),'./mnist/interpolation.png')
    elif DATASET ==1:
        save_image(interpolation_sample.data.view(num_interpolation_points, 3, 32, 32),'./CIFAR10/interpolation.png')
    elif DATASET ==2:
        save_image(interpolation_sample.data.view(num_interpolation_points, 3, 100, 100),'./cifar10_processed/interpolation.png')

if __name__ == '__main__':
    """
    DATASET takes values 0,1,2
    0: MNIST
    1: CIFAR10
    2: CIFAR10 + Stanford Cars
    """
    try:
        DATASET = int(sys.argv[1])
    except ValueError:
        raise
    train_loader, test_loader = load_data()
    model, optimizer = load_model_and_optimizer()
    train_and_test(model, optimizer, train_loader, test_loader)
    interpolate_images(model)
