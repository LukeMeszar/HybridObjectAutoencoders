import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

class VAE_CIFAR_v2(nn.Module):
    def __init__(self, ZDIMS, BATCH_SIZE, no_of_sample):
        super(VAE_CIFAR_v2, self).__init__()
        self.ZDIMS = ZDIMS
        self.BATCH_SIZE = BATCH_SIZE
        self.no_of_sample = no_of_sample
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), padding=(15, 15),
                               stride=2)  # This padding keeps the size of the image same, i.e. same padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), padding=(15, 15), stride=2)
        self.fc11 = nn.Linear(in_features=128 * 46 * 46, out_features=64)
        self.fc12 = nn.Linear(in_features=64, out_features=self.ZDIMS)
        self.fc21 = nn.Linear(in_features=128 * 46 * 46, out_features=64)
        self.fc22 = nn.Linear(in_features=64, out_features=self.ZDIMS)
        self.relu = nn.ReLU()
        # For decoder
        self.fc1 = nn.Linear(in_features=self.ZDIMS, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=25 * 25 * 128)
        self.conv_t1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, padding=1, stride=2)



    def encode(self, x: Variable) -> (Variable, Variable):
        x = x.view(-1, 3, 100, 100)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 128 * 46 * 46)
        mu_z = F.elu(self.fc11(x))
        mu_z = self.fc12(mu_z)
        logvar_z = F.elu(self.fc21(x))
        logvar_z = self.fc22(logvar_z)
        return mu_z, logvar_z


    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        if self.training:
            sample_z = []
            for _ in range(self.no_of_sample):
                std = logvar.mul(0.5).exp_()  # type: Variable
                eps = Variable(std.data.new(std.size()).normal_())
                sample_z.append(eps.mul(std).add_(mu))
            return sample_z
        else:
            return mu

    def decode(self, z: Variable) -> Variable:
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 128, 25, 25)
        x = F.relu(self.conv_t1(x))
        x = F.sigmoid(self.conv_t2(x))
        return x.view(-1, 3, 100*100)

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, 3, 100*100))
        z = self.reparameterize(mu, logvar)
        if self.training:
            return [self.decode(z) for z in z], mu, logvar
        else:
            return self.decode(z), mu, logvar
        
    def loss_function(self, recon_x, x, mu, logvar) -> Variable:
        if self.training:
            BCE = 0
            for recon_x_one in recon_x:
                BCE += F.binary_cross_entropy(recon_x_one, x.view(-1, 3, 100*100))
            BCE /= len(recon_x)
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3, 100*100))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.BATCH_SIZE * 100*100
        return BCE + KLD