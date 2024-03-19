# CS6910_Assignment_1


## Problem Statement
This assignment's objectives are to create our own feedforward neuram network and backpropagation code, apply gradient descent (and its variations SGD, Momentum, Nesterov, RMSprop, Adam, Nadam) using backpropagation, create our own optimizers, apply it to a classification job (of fashion_mnist dataset), and use [wandb.ai](url) to log our trials.

## Prerequisites
python 3.10
numpy 1.26.4
keras #ONLY FOR IMPORTING DATASET
Clone/download this repository
I have conducted all my experiments in Google Collab, for running in google colab, install wandb using following command -
!pip install wandb 
For running locally, install wandb and other required libraries using following command
```
pip install wandb
pip install numpy
pip install keras
```
## Dataset
I have used Fashion-MNIST dataset for complete experiments.
I have used MNIST dataset for Q10.
## Hyperparameters used in experiments
Sr. no	Hyperparameter	Variation/values used
1.	Activation function	Sigmoid, tanh,ReLu
2.	Loss function	Cross entropy, Mean squared error
3.	Initialisation	Random, Xavier
4.	Optimizer	Stochastic gradient descent, Momentum gradient descent, Nesterov gradient descent, RMSprop, ADAM, NADAM
5.	Batch size	32, 64 ,128
6.	Hidden layers	[64,64,64],[128,128,128],[256,256,256],[64,64,64,64],[64,64,64,64,64],[128,128,128,128],[128,128,128,128,128]
7.	Epochs	10,20,30
8.	Learning rate	0.001,0.0001
9.	Weight decay	0, 0.0005
10.	Dropout rate	0, 0.1
Question 1, 2, 3
The code for question 1,2,3 can be found here.

All experiments
All experiments are logged in this google colab notebook. (If having difficulty to view the code of colab notebook, refer .py file of the same here)

## Evaluation file(train.py)
For evaluating model download train.py file. (make sure you have all the prerequisite libraries installed).

And run the following command in the command line(this will take the default arguments).

python train.py 
The default evaluation run can be seen here in wandb.

The arguments supported by train.py file are:


# Code Specifications

Please ensure you add all the code used to run your experiments in the GitHub repository.

You must also provide a python script called `train.py` in the root directory of your GitHub repository that accepts the following command line arguments with the specified values -  

We will check your code for implementation and ease of use. We will also verify your code works by running the following command and checking wandb logs generated -

```
python train.py --wandb_entity myname --wandb_project myprojectname
```

### Arguments to be supported

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

<br>

**Please set the default hyperparameters to the values that give you your best validation accuracy.** (Hint: Refer to the Wandb sweeps conducted.)

You may also add additional arguments with appropriate default values.




Name	Default Value	Description
--wandb_project	"CS-6910 A1"	Project name used to track experiments in Weights & Biases dashboard
--wandb_entity	"shreyashgadgil007"	Wandb Entity used to track experiments in the Weights & Biases dashboard.
--dataset	"fashion_mnist"	choices: ["mnist", "fashion_mnist"]
--epochs	30	Number of epochs to train neural network.
--batch_size	32	Batch size used to train neural network.
--loss_function	"cross_entropy"	choices: ["square_error", "cross_entropy"]
--optimiser	"nadam"	choices: ["gd", "mgd", "ngd", "rmsprop", "adam", "nadam"]
--learning_rate	0.0001	Learning rate used to optimize model parameters
--weight_decay	0.0005	Weight decay used by optimizers.
--initialisation	"xavier"	choices: ["random", "xavier"]
--hidden_layer	[256,256,256]	Number of hidden layers used in feedforward neural network.
--activation	sigmoid	choices: ["sigmoid", "tanh", "relu"]
--dropout_rate	0.1	choice in range (0,1)
Supported arguments can also be found by:

python train.py -h
The default run has 5 epochs and hidden layer size [16,16,16]. Hence, it may have less acurate. Change the values if needed.

## Report
The wandb report for this assignment can be found here.

## Author
Vrijesh Kunwar ED23D015
