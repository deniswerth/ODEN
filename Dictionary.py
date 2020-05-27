import tensorflow as tf
import numpy as np

#Random seed initialization
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

class Dictionary():
    """
    This class defines all the neural net parameters
    including hyperparameter initializers, activation functions 
    and optimizers for the training. 
    """
    
    def __init__(self):

        self.Dict_initializers = {"GlorotNormal"  : "GlorotNormal",
                                  "GlorotUniform" : "GlorotUniform",
                                  "Ones"          : "Ones",
                                  "RandomNormal"  : "RandomNormal",
                                  "RandomUniform" : "RandomUniform",
                                  "Zeros"         : "Zeros"}
        
        self.Dict_activations = {"relu"         : tf.nn.relu,
                                 "sigmoid"      : tf.nn.sigmoid,
                                 "tanh"         : tf.nn.tanh,
                                 "elu"          : tf.nn.elu,
                                 "hard sigmoid" : tf.keras.activations.hard_sigmoid,
                                 "linear"       : tf.keras.activations.linear,
                                 "selu"         : tf.keras.activations.selu,
                                 "softmax"      : tf.keras.activations.softmax}
        
        self.Dict_optimizers = {"Adadelta" : tf.keras.optimizers.Adadelta(learning_rate = 0.001, rho = 0.95, epsilon = 1e-07),
                                "Adagrad"  : tf.keras.optimizers.Adagrad(learning_rate = 0.001, initial_accumulator_value = 0.1, epsilon = 1e-07),
                                "Adam"     : tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False),
                                "Adamax"   : tf.keras.optimizers.Adamax(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07),
                                "Nadam"    : tf.keras.optimizers.Nadam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07),
                                "RMSprop"  : tf.keras.optimizers.RMSprop(learning_rate = 0.001, rho = 0.9, momentum = 0.0, epsilon = 1e-07, centered = False),
                                "SGD"      : tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.0, nesterov = False)}
        
        self.Dict = {"initializer" : self.Dict_initializers, "activation" : self.Dict_activations, "optimizer" : self.Dict_optimizers}


