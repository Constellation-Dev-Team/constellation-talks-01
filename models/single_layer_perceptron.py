import numpy as np
import pandas as pd

from .activation_functions import ACTIVATION_FUNCTIONS, ACTIVATION_FUNCTIONS_GRADIENTS

class SingleLayerPerceptron:
    learning_rate = 0.01
    max_iter_num = 30000
    min_loss_error = 0.01

    
    def __init__(self, input_size, layer_size=10, ouput_size=1, hidden_activation_function='relu'):
        self.n_features = input_size
        self.hidden_layer_size = layer_size

        self.hidden_weights = np.random.uniform(0.8, 1.2, (input_size, layer_size))
        self.hidden_bias = np.ones((1, layer_size))

        self.output_weights = np.random.uniform(0.5, 1., (layer_size, 1))
        self.output_bias = np.ones((1, 1))
        
        self.saved_errors = []
        if not hidden_activation_function in ACTIVATION_FUNCTIONS:
            raise Exception('Função não suportada')
        self.hidden_activation_function_name = hidden_activation_function
        self.hidden_activation_function = ACTIVATION_FUNCTIONS[hidden_activation_function]
        self.hidden_activation_function_grad = ACTIVATION_FUNCTIONS_GRADIENTS[hidden_activation_function]
    
    def train(self, X, y):
        iter_num = 0
        delta2 = 10
        # try:
        while np.abs(np.mean(delta2)) > self.min_loss_error and iter_num < self.max_iter_num:
            iter_num += 1

            h0 = self.predict(X)
            y = np.array(y)
            delta2 = (h0 - y.reshape(h0.shape))

            dJW2 = self.A1.T @ delta2
            dJb2 =  np.sum(delta2, axis=0, keepdims=True)
            # Compute the gradients of the hidden layer
            grad = self.hidden_activation_function_grad(self.A1)
            delta1 = grad * (delta2 @ self.output_weights.T)
            JW1 =  X.T @ delta1
            Jb1 = np.sum(delta1, axis=0, keepdims=True)

            self.hidden_weights -= self.learning_rate * JW1
            self.hidden_bias -= self.learning_rate * Jb1
            self.output_weights -= self.learning_rate * dJW2
            self.output_bias -= self.learning_rate * dJb2
            
            e = delta2.mean()
            self.saved_errors.append(e)
            print(f"Época: {iter_num} Erro: {e}")
        # except:
        #     print('Erro no treinamento dos dados')
        return iter_num, e
    
    def predict(self, X):
        self.Z1 = (X@self.hidden_weights) + self.hidden_bias
        self.A1 = self.hidden_activation_function(self.Z1)

        self.Z2 = (self.A1@self.output_weights) + self.output_bias
        self.A2 = self.Z2
        return self.A2
        
    def predict_single(self, x):
        return self.predict(np.array([x]))