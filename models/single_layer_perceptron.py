import numpy as np
import pandas as pd

from .activation_functions import ACTIVATION_FUNCTIONS, ACTIVATION_FUNCTIONS_GRADIENTS

class SingleLayerPerceptron:
    weights = []
    learning_rate = 0.01
    
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

    def get_act_func_gradient(self, X):
        if self.hidden_activation_function_name == 'identity':
            return np.ones(X.shape)
        elif self.hidden_activation_function_name == 'relu':
            return self.hidden_activation_function_grad(self.A1)
        elif self.hidden_activation_function_name == 'sigmoid':
            return np.multiply(self.A1, 1 - self.A1)
        raise Exception('Função não suportada')
    
    def train(self, X, y):
        iter_num = 0
        loss = 10
        # try:
        while np.abs(np.mean(loss)) > 0.01 and iter_num < 30000:
            iter_num += 1

            h0 = self.predict(X)
            y = np.array(y)
            loss = (h0 - y.reshape(h0.shape))

            JWo = self.A1.T @ loss
            Jbo =  np.sum(loss, axis=0, keepdims=True)
            # Compute the gradients of the hidden layer
            Eh = np.multiply(self.get_act_func_gradient(self.A1), (loss @ self.output_weights.T))
            JWh =  X.T @ Eh
            Jbh = np.sum(Eh, axis=0, keepdims=True)

            self.hidden_weights -= self.learning_rate * JWh
            self.hidden_bias -= self.learning_rate * Jbh
            self.output_weights -= self.learning_rate * JWo
            self.output_bias -= self.learning_rate * Jbo
            
            e = loss.mean()
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