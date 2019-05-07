import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim
import sys
import os
import argparse
from PIL import Image

sys.path.append(os.path.join(os.getcwd(),'autoencoders'))
import STL_v1 as S
import STL_v2 as SU
import CIFAR_v3 as C
import CIFAR_unet as CU
import CARS as CA
import MNIST

plt.style.use('ggplot')

def parse_args():
    parser = argparse.ArgumentParser(description='Hybridize Two Opjects: This script requires that you have the data'
    'in /data and are using images from the set the model was trained on.')
    parser.add_argument('-m', metavar='Model_Path', type=str,
                    help='path to torch model to use', required=True)
    parser.add_argument('-i1', metavar='First_Image_Path', type=str,
                    help='path to fist image', required=True)
    parser.add_argument('-i2', metavar='Second_Image_Path', type=str,
                    help='path to second image', required=True)


    args = parser.parse_args()
    img = Image.open(args.i1)
    img_1 = np.array(img)
    img_1 = img_1/255.

    img = Image.open(args.i2)
    img_2 = np.array(img)
    img_2 = img_2/255.

    model_path = args.m

    return img_1, img_2, model_path

def show_transition(model, loader, device, img_1, img_2, n=10):
    encoded_vecs = torch.zeros([n, model.hidden_layer_size])

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            print(data[0].cpu().numpy())
            encoded = model.encoder(data)
            break

        first_img = img_1
        second_img = img_2

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
    img_1, img_2, model_path = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path)
    model.eval()

    if "model_stl.pth" in model_path:
        train_load, test_load = S.get_stl_loaders(download=False)
    elif "model_stl_u.pth" in model_path:
        train_load, test_load = SU.get_stl_loaders(download=False)
    elif "model_cifar_v1.pth" in model_path:
        train_load, test_load = C.get_stl_loaders(download=False)
    elif "model_stl_u.pth" in model_path:
        train_load, test_load = CU.get_stl_loaders(download=False)
    else:
        print("Given model isn't a provided pre trained model, run download_models.py and run again")
        exit(0)

    show_transition(model, train_load, device, img_1, img_2)
