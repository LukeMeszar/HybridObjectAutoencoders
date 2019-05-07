# Autoencoders

This section of the project contains the various autoencoder-based attempts to create hybrid objects.

# Contents

## CARS dataset:
The CARS dataset contains background-filtered images of cars, cats, dogs, and horses.

- `cars-setup.sh`: a shell script that sets up data for the CARS files.
- `CARS.py`: Python file that contains everything needed to run the CARS model
- `cars.ipynb`: Jupyter notebook that contains logic for running the CARS model
- How to run:
  - Run `bash cars-setup.sh` to download and extract data
  - Run `python3 CARS.py` to train the model and display the results
  - Alternatively, open `cars.ipynb` with Jupyter, which offers similar functionality as the latter
   
## CIFAR dataset:
For the CIFAR-10 dataset, we made several different models.

- CIFAR V1 is a basic convolutional neural network, with the encoder and decoder trained simultaneously.
  - `CIFAR_v1.py`: Python file that contains all required logic to run this model
  - How to run:
    - Run `python3 CIFAR_v1.py` to download data, train the model, and display the results.
- CIFAR V2 is the best-working model. For this model, we train the encoder (as a classifier, trying to separate the classes in encoded space) and decoder separately.
  - `cifar-get-pretrained-models.sh`: a shell script that will download all of our pretrained models for CIFAR V2
  - `CIFAR_v2.py`: Python file that contains everything needed to run the CIFAR V2 model
  - `CIFAR-separate-training.ipynb`: Jupyter notebook that contains logic for running the CIFAR V2 model
  - How to run:
    - Run `bash cifar-get-pretrained-models.sh` to download the pretrained models
    - Run `python3 CIFAR_v2.py` to download data, train the model, and display the results.
    - Alternatively, run `python3 CIFAR_v2.py <model_name_here>` to display the results of the model given the pretrained model `<model_name_here>`
      - For example, `python3 CIFAR_v2.py cifar-v2-model-4.pth`
    - Alternatively, use the Jupyter notebook to run.
- CIFAR V3 is a model that optimizes the performance of the classifier. Surprisingly, it did not work well.
  - The idea behind this model is that if the encoder can easily separate all of the classes, the encoded space is more structured and therefore easier to decode.
  - `CIFAR_v3.py`: Python file that contains everything needed to run the CIFAR V3 model
  - `CIFAR-classifier.ipynb`: Jupyter notebook that contains logic for running the CIFAR V3 model
  - How to run:
    - Run `python3 CIFAR_v3.py` to download data, train the model, and display the results.
    - Alternatively, use the Jupyter notebook to run.
- CIFAR U-Net is a U-Net based model on the CIFAR-10 dataset. It didn't work well, not because the autoencoder didn't recreate the images well (it did a great job with recreation) but because the transitions happened more in data space than in encoded space: objects simply faded in and out, which is precisely what this project is trying to avoid.
  - `CIFAR_unet.py`: Python file that contains everything needed to run the CIFAR U-Net model
  - `CIFAR-unet.ipynb`: Jupyter notebook that contains logic for running the CIFAR U-Net model
  - How to run:
    - Run `python3 CIFAR_unet.py` to download data, train the model, and display the results.
    - Alternatively, use the Jupyter notebook to do the same.

## MNIST dataset:
For the MNIST dataset, we only made one model. It works really well.

- `MNIST.py`: Python file that contains everything needed to run the MNIST model
- How to run:
  - Run `python3 MNIST.py` to download data, train the model, and display the results.
  
## STL-10 dataset:
For the STL-10 dataset, which contains 96x96 images, like CIFAR, of 10 different classes, we created two different models: one based on standard convolutional autoencoders and the other based on a U-Net.

- STL V1 is the standard convolutional autoencoder model. It tends to result in the correct general shape in its recreations, but lack details.
  - `stl-get-pretrained-models.sh`: a shell script that downloads both of our pretrained STL models.
  - `STL_v1.py`: Python file that contains everything needed to run the STL V1 model
  - `STL_v1.ipynb`: Jupyter notebook that contains logic for running the STL V1 model
  - How to run:
    - Run `bash stl-get-pretrained-models.sh` to download the pretrained models
    - Run `python3 STL_v1.py` to download training data, train the model, and display the results.
    - Run `python3 STL_v1.py <model_name_here>` to display the results of of the model given the pretrained model `<model_name_here>`
    - Alternatively, all of this functionality is available in the Jupyter notebook.
- STL V2 is the U-Net architecture applied to the STL dataset. It is quite blurry, moreso than V1. We assumed that adding skip connections would increase the resolution, but it did not.
  - `stl-get-pretrained-models.sh`: a shell script that downloads both of our pretrained STL models.
  - `STL_v2.py`: Python file that contains everything needed to run the STL V2 model
  - `STL_unet.ipynb`: Jupyter notebook that contains logic for running the STL V2 model
  - How to run:
    - Run ``bash stl-get-pretrained-models.sh` to download the pretrained models
    - Run `python3 STL_v2.py` to download training data, train the model, and display the results.
    - Run `python3 STL_v2.py <model_name_here>` to display the results of of the model given the pretrained model `<model_name_here>`
    - Alternatively, all of this functionality is available in the Jupyter notebook.
