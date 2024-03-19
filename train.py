"""
@author: Vrijesh Kunwar
"""

from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
import wandb
import warnings
import argparse

wandb.login(key='3c21150eb43b007ee446a1ff6e87f640ec7528c4')

def to_categorical(labels, num_classes):
    # Initialize an array of zeros with shape (num_classes, num_samples)
    one_hot = np.zeros((num_classes, labels.shape[0]))
    # Set the corresponding index of each sample to 1
    one_hot[labels, np.arange(labels.shape[0])] = 1
    return one_hot.T  # Transpose the array to match the desired shape


def dataset_preprocess(dataset):
    #change the code below to accept different dataset

    if dataset=="mnist":
        (train_ima, train_lab), (test_ima, test_lab) = mnist.load_data()
    else:
        (train_ima, train_lab), (test_ima, test_lab) = fashion_mnist.load_data()  

    # Preprocess the data
    train_images = train_ima / 255.0
    test_images = test_ima / 255.0
    train_images = train_images.reshape(train_images.shape[0], -1).T
    test_images = test_images.reshape(test_images.shape[0], -1).T
    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_lab, num_classes=10).T
    test_labels = to_categorical(test_lab, num_classes=10).T

    # Split train set into train and validation
    val_split = int(train_images.shape[1] * 0.1)
    val_images = train_images[:, :val_split]
    val_labels = train_labels[:, :val_split]
    train_images = train_images[:, val_split:]
    train_labels = train_labels[:, val_split:]

    #return train_images,X_test.T,val_images,train_labels ,val_labels,Y_test.T,Y_train_encoded.T,Y_val_encoded.T,Y_test_encoded.T
    return train_images,val_images,train_labels ,val_labels


class NurlNtwk:
    def __init__(self, input_layer, hidden_layers, output_layer, initialization, activation, loss_function, optimizer, dropout_rate=0):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.initialization = initialization
        self.activation = activation
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.parameters = None

    def initialize_parameters(self):
        np.random.seed(1)
        parameters = {}
        layer_dims = [self.input_layer] + self.hidden_layers + [self.output_layer]
        L = len(layer_dims)
        for l in range(1, L):
            if self.initialization == "random":
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            elif self.initialization == "Xavier":
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1 / layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        self.parameters = parameters

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def tanh(self, Z):
        return np.tanh(Z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_propagation(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2
        for l in range(1, L):
            A_prev = A
            Z = np.dot(self.parameters['W' + str(l)], A_prev) + self.parameters['b' + str(l)]
            if self.activation == "sigmoid":
                A = self.sigmoid(Z)
            elif self.activation == "relu":
                A = self.relu(Z)
            elif self.activation == "tanh":
                A = self.tanh(Z)
            cache = (A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], Z)
            caches.append(cache)
        ZL = np.dot(self.parameters['W' + str(L)], A) + self.parameters['b' + str(L)]
        AL = self.softmax(ZL)
        cache = (A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], ZL)
        caches.append(cache)
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        if self.loss_function == "cross_entropy":
            cost = -np.sum(np.multiply(Y, np.log(AL))) / m
        if self.loss_function == "squared_error":
            # Default to mean squared error if loss function is not specified or recognized
            cost = np.sum((AL - Y) ** 2) / (2 * m)
        return cost

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dZL = AL - Y
        A_prev, W, b, Z = caches[L-1]
        grads['dW' + str(L)] = np.dot(dZL, A_prev.T) / m
        grads['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m
        dAL = np.dot(W.T, dZL)
        for l in reversed(range(L-1)):
            A_prev, W, b, Z = caches[l]
            if self.activation == 'relu':
                dZ = dAL.copy()
                dZ[Z <= 0] = 0  # Derivative of ReLU
            elif self.activation == 'sigmoid':
                sigmoid_Z = self.sigmoid(Z)
                dZ = dAL * sigmoid_Z * (1 - sigmoid_Z)  # Derivative of sigmoid
            elif self.activation == 'tanh':
                tanh_Z = self.tanh(Z)
                dZ = dAL * (1 - tanh_Z**2)  # Derivative of tanh
            grads['dW' + str(l + 1)] = np.dot(dZ, A_prev.T) / m
            grads['db' + str(l + 1)] = np.sum(dZ, axis=1, keepdims=True) / m
            dAL = np.dot(W.T, dZ)
        return grads
    def initialize_optimizer(self):
        self.velocity = {}
        self.momentum = 0
        self.rmsprop_cache = {}
        self.adam_params = {'m': {}, 'v': {}}
        self.nadam_params = {'m': {}, 'v': {}}

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(L):
            if self.optimizer == "SGD":
                self.parameters['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
                self.parameters['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)]
            elif self.optimizer == "Momentum":
                v_dW = 0.9 * grads['dW' + str(l+1)]
                v_db = 0.9 * grads['db' + str(l+1)]
                self.parameters['W' + str(l+1)] -= learning_rate * v_dW
                self.parameters['b' + str(l+1)] -= learning_rate * v_db
            elif self.optimizer == "Nesterov":
                v_dW = 0.9 * grads['dW' + str(l+1)]
                v_db = 0.9 * grads['db' + str(l+1)]
                self.parameters['W' + str(l+1)] -= learning_rate * (v_dW + 0.9 * v_dW)
                self.parameters['b' + str(l+1)] -= learning_rate * (v_db + 0.9 * v_db)
            elif self.optimizer == "RMSprop":
                s_dW = 0.99 * grads['dW' + str(l+1)]**2 + (1 - 0.99) * (grads['dW' + str(l+1)]**2)
                s_db = 0.99 * grads['db' + str(l+1)]**2 + (1 - 0.99) * (grads['db' + str(l+1)]**2)
                self.parameters['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)] / (np.sqrt(s_dW) + (1e-8))
                self.parameters['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)] / (np.sqrt(s_db) + (1e-8))
            elif self.optimizer == "Adam":
                v_dW = 0.9 * grads['dW' + str(l+1)] + (1 - 0.9) * grads['dW' + str(l+1)]
                v_db = 0.9 * grads['db' + str(l+1)] + (1 - 0.9) * grads['db' + str(l+1)]
                s_dW = 0.99 * grads['dW' + str(l+1)]**2 + (1 - 0.99) * (grads['dW' + str(l+1)]**2)
                s_db = 0.99 * grads['db' + str(l+1)]**2 + (1 - 0.99) * (grads['db' + str(l+1)]**2)
                self.parameters['W' + str(l+1)] -= learning_rate * v_dW / (np.sqrt(s_dW) + (1e-8))
                self.parameters['b' + str(l+1)] -= learning_rate * v_db / (np.sqrt(s_db) + (1e-8))
            elif self.optimizer == "Nadam":
                v_dW = 0.9 * grads['dW' + str(l+1)] + (1 - 0.9) * grads['dW' + str(l+1)]
                v_db = 0.9 * grads['db' + str(l+1)] + (1 - 0.9) * grads['db' + str(l+1)]
                s_dW = 0.99 * grads['dW' + str(l+1)]**2 + (1 - 0.99) * (grads['dW' + str(l+1)]**2)
                s_db = 0.99 * grads['db' + str(l+1)]**2 + (1 - 0.99) * (grads['db' + str(l+1)]**2)
                v_dW_corrected = v_dW / (1 - 0.9)
                v_db_corrected = v_db / (1 - 0.9)
                s_dW_corrected = s_dW / (1 - 0.99)
                s_db_corrected = s_db / (1 - 0.99)
                self.parameters['W' + str(l+1)] -= learning_rate * v_dW_corrected / (np.sqrt(s_dW_corrected) + (1e-8))
                self.parameters['b' + str(l+1)] -= learning_rate * v_db_corrected / (np.sqrt(s_db_corrected) + (1e-8))

    def train(self, X_train, Y_train, X_val, Y_val, epochs, learning_rate,batch_size, beta1=0.9, beta2=0.999, epsilon=1e-8, WandB=False):
        train_costs = []
        val_costs = []
        val_acc = []
        train_acc = []
        self.initialize_parameters()

        for epoch in range(int(epochs)):
          for batch in range(0, X_train.shape[1], batch_size):

                batch_images =  X_train[:,batch:batch+batch_size]
                batch_output =  train_labels[:,batch:batch+batch_size]
                AL, caches =  self.forward_propagation(batch_images)
                grads = self.backward_propagation(AL, batch_output, caches)

                if self.optimizer == "SGD":
                    self.update_parameters(grads, learning_rate)
                elif self.optimizer == "Momentum":
                    self.update_parameters(grads, learning_rate)
                elif self.optimizer== "Nesterov":
                    self.update_parameters(grads, learning_rate)
                elif self.optimizer == "RMSprop":
                    self.update_parameters(grads, learning_rate)
                elif self.optimizer == "Adam":
                    self.update_parameters(grads, learning_rate)
                elif self.optimizer == "Nadam":
                    self.update_parameters(grads, learning_rate)

          train_predictions = self.predict(X_train)
          val_predictions = self.predict(X_val)
          train_acc = self.accuracy(train_predictions, Y_train)
          val_acc = self.accuracy(val_predictions, Y_val)

          print(f"Epoch {epoch+1}/{epochs}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")

          if WandB:
              wandb.log({"val_accuracy": val_acc,"accuracy": train_acc,"steps":epoch},)
        return train_acc

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return np.argmax(AL, axis=0)

    def accuracy(self, predictions, labels):
        return np.mean(predictions == np.argmax(labels, axis=0))




if __name__ == '__main__':    
  
    parser = argparse.ArgumentParser(description='Train a neural network on the MNIST dataset.')
    parser.add_argument('--wandb_entity', type=str, default='ed23d015', help='Name of the wandb entity')
    parser.add_argument('--wandb_project', type=str, default='DL_Assignment_1', help='Name of the wandb project')
    parser.add_argument('--epochs', type=int, default=10, help='No. of epochs for the run')
    parser.add_argument('--batch_size', type=int, default=128, help='No. of batch size for the run')
    parser.add_argument('--learning_rate', type=int, default=0.001, help='Learning rate to optimize model')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', help='choices:["cross_entropy","square_error"]')
    parser.add_argument('--initialisation', type=str, default='Xavier', help='choices:["xavier","random"]')
    parser.add_argument('--optimiser', type=str, default='Nadam', help='choices:["gd","mgd","ngd","rmsprop","nadam","adam"]')
    parser.add_argument('--activation', type=str, default='tanh', help='choices:["tanh","sigmoid","relu"]')
    parser.add_argument('--weight_decay', type=int, default=0.0005, help='weight decay value, should be generally low')
    parser.add_argument('--dropout_rate', type=int, default=0, help='dropout value range: (0,1)')
    parser.add_argument('--hidden_layer', type=list, default=[128,128,128], help='No. of hidden layers, format should be in list like:[64,64,64,64],[64,64,64,64,64]') 
    parser.add_argument('--dataset', type=str, default='fashion_mnist', help='choices:["fashion_mnist"],["mnist"]') 

    args = parser.parse_args()

    train_images,val_images,train_labels ,val_labels = dataset_preprocess(args.dataset)
    sweep_config = {
        'method': 'bayes', #grid, random,bayes
        'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
        },
        'parameters': {
            'epochs': {
                'values': [args.epochs]
            },
            'learning_rate': {
                'values': [args.learning_rate]
            },
            'loss_function':{
                'values':[args.loss_function]
            },
            'initilisation':{
                'values':[args.initialisation]
            },
            'batch_size':{
                'values':[args.batch_size]
            },
            'optimiser': {
                'values': [args.optimiser]
            },
            'activation': {
                'values': [args.activation]
            },
            'hidden_layer': {
                'values': [
                            args.hidden_layer]
            },
            'dropout_rate':{
                'values':[args.dropout_rate]  
                    },
            
            'weight_decay':{
                'values':[args.weight_decay]
            }

        }
    }
    


    def evaluate():
        
        train_images,val_images,train_labels ,val_labels = dataset_preprocess(args.dataset)

        config_defaults = {
            'epochs': 10,
            'input_layer': 784,
            'output_layer': 10,
            'batch_size':128,
            'dropout_rate':0,
            'weight_decay':0.0005,
            'learning_rate': 0.001,
            'hidden_layer':[128,128,128],
            'optimiser':'Nadam',
            'activation':'tanh',
            'initialisation':'Xavier',
            'loss_function':'cross_entropy'

        }

        # New wandb run
        #wandb.init(project='DL_Assignment_1', entity='ed23d015',config=config_defaults)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,config= config_defaults)
        wandb.run.name = 'FinalEval_run(ED23D015_Vrijesh): '+'b_s:'+str(wandb.config.batch_size)+',lr:'+ str(wandb.config.learning_rate)+'drop:'+str(wandb.config.dropout_rate)+',ep:'+str(wandb.config.epochs)+ ',opt:'+str(wandb.config.optimiser)+ ',hl:'+str(wandb.config.hidden_layer)+ ',act:'+str(wandb.config.activation)+',decay:'+str(wandb.config.weight_decay)+',init:'+str(wandb.config.initialisation)+',loss:'+str(wandb.config.loss_function)


        config = wandb.config
        learning_rate = config.learning_rate
        epochs = int(config.epochs)
        hidden_layer = config.hidden_layer
        activation = config.activation
        optimiser = config.optimiser
        input_layer = config.input_layer
        output_layer = config.output_layer
        batch_size = config.batch_size
        weight_decay = config.weight_decay
        loss_function = config.loss_function
        initialisation = config.initialisation
        dropout_rate=config.dropout_rate
        # Training
        S_network    = NurlNtwk(input_layer, hidden_layer, output_layer,initialization=initialisation,activation=activation,loss_function=loss_function,optimizer=optimiser,dropout_rate=dropout_rate)
        acc1  = S_network.train(train_images, train_labels, val_images, val_labels, epochs=epochs, batch_size = batch_size, learning_rate=learning_rate, WandB=True)

    #sweep_id = wandb.sweep(sweep_config, entity='ed23d015', project="DL_Assignment_1")
    sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entity, project=args.wandb_project)
    
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    wandb.agent(sweep_id, function=evaluate, count=1)
    wandb.finish()
