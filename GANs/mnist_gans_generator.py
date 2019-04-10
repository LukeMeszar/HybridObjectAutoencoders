import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger
from mnist_gans import GeneratorNet, vectors_to_images, noise
from utils import Logger
import numpy as np
import os
import sys

if __name__ == '__main__':
    generator = GeneratorNet()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_state_path = dir_path + "/data/models/VGAN/MNIST/G_epoch_199"
    generator.load_state_dict(torch.load(model_state_path))
    logger = Logger(model_name='VGAN_test', data_name='MNIST')
    num_test_samples = 16
    test_noise = noise(num_test_samples)
    test_images = vectors_to_images(generator(test_noise))
    test_images = test_images.data
    logger.log_images(
        test_images, num_test_samples,
        1, 1, 1
    )
    test_1 = test_images[0].numpy()
    test_2 = test_images[-1].numpy()
    num_interpolation_points = 24
    interpolated_images = [test_1]
    for x in np.linspace(0,1, num_interpolation_points - 2):
        interpolated_image = test_1*x + test_2*(1-x)
        interpolated_images.append(interpolated_image)
    interpolated_images.append(test_2)
    interpolated_images = np.array(interpolated_images)
    interpolated_tensor = torch.Tensor(interpolated_images)
    logger.log_images(
        interpolated_tensor, num_interpolation_points,
        2,2,2
    )
