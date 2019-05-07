import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim
import sys
import argparse
import os

sys.path.append(os.path.join(os.getcwd(),'autoencoders'))
import STL_v1 as S
import STL_v2 as SU
import CIFAR_v3 as C
import CIFAR_unet as CU
import CARS as CA
import MNIST



plt.style.use('ggplot')

def parse_args_and_setup():
    parser = argparse.ArgumentParser(description='Train Model ')
    parser.add_argument('-STL',
                    help='train stl model', action='store_true')
    parser.add_argument('-STL_U',
                    help='train stl u net model', action='store_true')
    parser.add_argument('-CIFAR',
                    help='train cifar model', action='store_true')
    parser.add_argument('-CIFAR_U',
                    help='train cifar u net model', action='store_true')
    parser.add_argument('-MNIST',
                    help='train mnist model', action='store_true')
    parser.add_argument('-CARS',
                    help='train cars model', action='store_true')
    parser.add_argument('-ALL',
                    help='train all models', action='store_true')
    parser.add_argument('-o', metavar='Model_Output_Path', type=str,
                    help='path of output model', required=True)

    args = parser.parse_args()
    output_path = args.o

    if args.STL:
        return "stl", output_path
    if args.STL_U:
        return "stl_u", output_path
    if args.CIFAR:
        return "cifar", output_path
    if args.CIFAR_U:
        return "cifar_u", output_path
    if args.MNIST:
        return "mnist", output_path
    if args.CARS:
        return "cars", output_path
    if args.ALL:
        return "all", output_path
    else:
        print("no model indicated")
        exit(0)

def train_stl(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = S.STLAutoEncoder(180).to(device)
    epoch = 0
    train_load, test_load = S.get_stl_loaders(download=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(50):
        S.train(model, device, train_load, opt, epoch)
        S.test(model, device, train_load)
        epoch += 1
    S.show_transition(model, train_load, device)
    torch.save(model, os.path.join(model, "model_stl.pth"))

def train_stl_u(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SU.STLAutoEncoder(128, 32).to(device)
    epoch = 0
    train_load, test_load = SU.get_stl_loaders(download=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(50):
        SU.train(model, device, train_load, opt, epoch)
        SU.test(model, device, train_load)
        epoch += 1
    SU.show_transition(model, train_load, device)
    torch.save(model, os.path.join(model, "model_stl_u.pth"))

def train_cifar(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = C.CIFARAutoEncoderV3(80).to(device)
    epoch_enc = 0
    epoch_dec = 0
    train_load, test_load = C.get_cifar_loaders(download=True)
    opt_enc = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=0.001)
    for _ in range(50):
        epoch_dec += 1
        C.train_decoder(model, device, train_load, opt_dec, epoch_dec)
        C.test_decoder(model, device, test_load)
    C.show_transition(model, train_load, device)
    torch.save(model, os.path.join(model, "model_cifar.pth"))

def train_cifar_u(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CU.CIFARAutoEncoderUNet(20).to(device)
    train_load, test_load = CU.get_cifar_loaders(download=True)
    epoch = 0
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(50):
        CU.train(model, device, train_load, opt, epoch)
        CU.test(model, device, train_load)
        epoch += 1
    CU.show_transition(model, train_load, device)
    torch.save(model, os.path.join(model_path, "model_cifar_u.pth"))

def train_mnist(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mnist_model = MNIST.MnistAutoEncoder(255).to(device)
    mnist_epoch = 0
    mnist_train_loader, mnist_test_loader = MNIST.get_mnist_loaders()
    mnist_opt = torch.optim.SGD(mnist_model.parameters(), lr=0.01, momentum=0.5)
    for _ in range(10):
        mnist_epoch += 1
        MNIST.train(mnist_model, device, mnist_train_loader, mnist_opt, mnist_epoch)
        MNIST.test(mnist_model, device, mnist_test_loader)

    MNIST.show_transition(mnist_model, mnist_test_loader, device)
    torch.save(mnist_model, os.path.join(model_path, "model_mnist.pth"))

def train_cars(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CA.CARSAutoEncoderV3(512).to(device)
    train_load, test_load = CA.get_cars_loaders(download=True)
    opt_enc = torch.optim.Adam(model.classifier.parameters(), lr=0.01)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=0.001)

    epoch_enc = 0
    for _ in range(120):
        CA.train_encoder(model, device, train_load, opt_enc, epoch_enc)
        CA.test_encoder(model, device, test_load)
        epoch_enc += 1

    epoch_dec = 0
    for _ in range(50):
        CA.train_decoder(model, device, train_load, opt_dec, epoch_dec)
        CA.test_decoder(model, device, test_load)
        epoch_dec += 1

    CA.show_transition(model, train_load, device)
    torch.save(model, os.path.join(model_path, "model_cars.pth"))

if __name__ == "__main__":
    model, model_path = parse_args_and_setup()
    if model == "stl":
        train_stl(model_path)
    if model == "stl_u":
        train_stl_u(model_path)
    if model == "cifar":
        train_cifar(model_path)
    if model == "cifar_u":
        train_cifar_u(model_path)
    if model == "mnist":
        train_mnist(model_path)
    if model == "cars":
        train_cars(model_path)
    if model == "all":
        train_stl(model_path)
        train_stl_u(model_path)
        train_cifar(model_path)
        train_cifar_u(model_path)
        train_mnist(model_path)
        train_cars(model_path)
