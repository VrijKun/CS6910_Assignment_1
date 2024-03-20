# CS6910_Assignment_1


## Problem Statement
This assignment's objectives are to create our own feedforward neuram network and backpropagation code, apply gradient descent (and its variations SGD, Momentum, Nesterov, RMSprop, Adam, Nadam) using backpropagation, create our own optimizers, apply it to a classification job (of fashion_mnist dataset), and use [wandb.ai](wandb.ai) to log our trials.

## Prerequisites
```
python 3.10
numpy 1.26.4
keras #ONLY FOR IMPORTING DATASET
```
Clone/download this repository.
All my experiments have been performed on Google Collab, to run it on google colab, install wandb from the following command -
```
!pip install wandb 
```
Use the following command to install wandb and other necessary libraries for local operation.
```
pip install wandb
pip install numpy
pip install keras
```
## Dataset
Fashion-MNIST dataset have been used to complete experiments.
MNIST dataset have been used for Q10.
## Hyperparameters
| Sr. no |	Hyperparameter |	Variation/values used |
| ------ |	------------- | --------------------- |
| 1. |	Activation function |	Sigmoid, tanh,ReLu |
| 2. |	Loss function |	Cross entropy, Mean squared error |
| 3. |	Initialisation |	Random, Xavier |
| 4. |	Optimizer |	Stochastic gradient descent, Momentum gradient descent, Nesterov gradient descent, RMSprop, ADAM, NADAM |
| 5. |	Batch size |	32, 64 ,128 |
| 6. |	Hidden layers |	[64,64,64],[128,128,128],[32,32,32],[64,64,64,64],[64,64,64,64,64],[128,128,128,128],[128,128,128,128,128] |
| 7. |	Epochs |	5,10 |
| 8. |	Learning rate |	0.001,0.0001 |
| 9. |	Weight decay |	0, 0.0005, 0.5 |
| 10. |	Dropout rate |	0 |

## Solutions of all the Questions
The solution code for all the experiments can be found [Here](https://github.com/VrijKun/CS6910_Assignment_1/blob/main/Assignment_1_DL_ED23D015.ipynb). It has different experiments to solve all the questions.

## 2nd experiments
This contains the experiments results for Que 4, Que 5, Que 6.
Experiments are available in [google colab notebook](https://github.com/VrijKun/CS6910_Assignment_1/blob/main/Assignment_1_DL_ED23D015_2nd_Experimants.ipynb).

Just in case train.py file is [here](https://github.com/VrijKun/CS6910_Assignment_1/blob/main/train.py)

## Evaluation file(train.py)
For evaluating the model download [train.py](https://github.com/VrijKun/CS6910_Assignment_1/blob/main/train.py) file. (You will need to install the prerequisite libraries beforehand).

And run the following command in the command line(this will take the default arguments).

## Code Specifications
following command line arguments with the specified values available in train.py file are:

And run the following command in the command line(this will take the default arguments)

```
python train.py
```

## Arguments to be supported

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | DL_Assignment_1 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | ed23d015  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 128 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | Nadam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-w_d`, `--weight_decay` | .0005 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | Xavier | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 3 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128,128,128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["sigmoid", "tanh", "ReLU"] |

<br>


python train.py -h
The default run has 128 batch size and hidden layer size [128,128,128]. Hence, it may be little slow. Change the values if needed.

## Report
The wandb report for this assignment can be found [here.](https://wandb.ai/ed23d015/DL_Assignment_1/reports/ED23D015-CS6910-Assignment-1--Vmlldzo3MDU5NTQ3)

## Author
Vrijesh Kunwar ED23D015
