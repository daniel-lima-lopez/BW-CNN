# BW-CNN
This repository contains the code for the experiments described in the article **"Butterworth CNN: an improvement on memory use for Fourier Convolutional Neural Networks"** authored by Daniel Lima-López and [Pilar Gómez-Gil](https://scholar.google.com/citations?user=M3yVI1oAAAAJ&hl=es), to be published in the proceedings of the 2024 IEEE PES Generation, Transmission, and Distribution & IEEE Autumn Meeting on Power, Electronics and Computing Joint Conference. The code features the implementation of the proposed BW-CNN architecture, along with replicas of the [SB-CNN](https://www.sciencedirect.com/science/article/abs/pii/S0925231219310148) and conventional CNN architectures, which are used for comparing the performance of the proposed method.

This work received the **Best Computing Track Paper Award**:

<img src="ROPEC.jpeg">

## Installation
The implementation was carried out using Keras, specifically with TensorFlow version 2.10 and Python version 3.9.19.

Regarding the installation, the project should be cloned as follows:
```bash
git clone git@github.com:daniel-lima-lopez/BW-CNN.git
```
Afterwards, you can try the code in the downloaded folder:
```bash
cd BW-CNN
```

## Usage
The [FourierNetworks.py](FourierNetworks.py) file includes the implementation of all the Fourier Networks tools used in the paper, including: convolutional layers `Dot` and `ButterworthLayer` used for SB-CNN and BW-CNN, respectively, the `RandomLowHigh` class for defining the auxiliary parameters `A` and `b` in Butterworth filters, the `Spect_Avg_Pool` class for the new pooling layer, the activation function `CReLU` and the `IFFT` transformation layer.

Note that, since the implemented code was made considering the base classes of keras, then it can be used in conjunction with any other element of keras, as long as the output dimension of the tensors allows it.

Notebooks [CNN.ipynb](CNN.ipynb), [SB-CNN.ipynb](SB-CNN.ipynb) and [BW-CNN.ipynb](BW-CNN.ipynb) include examples of the execution of the architectures used in the experiments with the [Colorectal Histology](https://www.tensorflow.org/datasets/catalog/colorectal_histology) dataset (HMNIST). Note that the class `MaxEpoch`, necessary to detect the optimal epoch for the evaluation on the test, set is included. Moreover, in the case of SB-CNN and BW-CNN, the preprocessing performed to work with the centered frequency representation of the images is included.
