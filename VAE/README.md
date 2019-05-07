# VAE

This section of the project contains the various VAE-based attempts to create hybrid objects.

# Datasets

A VAE model was created for three different datasets.

## MNIST

The standard dataset MNIST consisting of 28x28 images of handwritten digits.
This is represented by DATASET 0 when running the script.

## CIFAR10

The standard dataset CIFAR10 consisting of 32x32 images in ten classes of objects.
This is represented by DATASET 1 when running the script.

## CIFAR10 + CARS
The CIFAR10 + CARS dataset contains 50x50 background-filtered images of cars, cats, dogs, and horses.
The cars are from the Stanford Cars dataset and the other three categories are from CIFAR10.
This is represented by DATASET 2 when running the script.

## Running models

To run the different models, run *vae.sh $1* where $1 is either 0,1, or 2.
This corresponds to the datasets MNIST, CIFAR10, and CIFAR10 + CARS above.
The appropriate data will be automatically downloaded into the appropriate directories.
Then, CNN_VAE.py will run which will train and test the appropriate models.
