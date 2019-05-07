# Autoencoders

This section of the project contains the various autoencoder-based attempts to create hybrid objects.

# Contents

## CARS dataset:
The CARS dataset contains background-filtered images of cars, cats, dogs, and horses.
We made two models for the CARS dataset, one based on a standard convolutional neural network
and one based on a U-Net convolutional neural network.

- `cars-setup.sh`: a shell script that sets up data for the CARS files.
- `CARS.py`: Python file that contains everything needed to run the CARS model
- `cars.ipynb`: Jupyter notebook that contains logic for running the CARS model
- How to run:
  - Run `sh cars-setup.sh` to download and extract data
  - Run `python3 CARS.py` to train the model and display the results
  - Alternatively, open `cars.ipynb` with Jupyter, which offers similar functionality as the latter
- `CARS_unet.py`: U-Net based model for the CARS dataset
- `cars_unet.ipynb`: Jupyter notebook that contains logic for running the CARS U-Net model
- How to run (U-Net):
  - Run `sh cars-setup.sh` to download and extract data (only need to do once, total)
  - Run `python3 CARS_unet.py` to train the model and display the results
  - Alternatively, open `cars_unet.ipynb` with Jupyter, which offers similar functionality as the latter
   
## CIFAR dataset:
For the CIFAR-10 dataset, we made several different models.
