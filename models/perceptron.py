import numpy as np
from .activation_functions import ACTIVATION_FUNCTIONS, ACTIVATION_FUNCTIONS_GRADIENTS

class Perceptron:
    learning_rate = 0.01
    max_iter_num = 10000
    min_loss_error = 0.01

    def __init__(self, size, activation_function='identity'):
        self.n_features = size
        self.weights = np.ones(size) * 1/size
        self.saved_errors = []
        self.bias = 0
        if activation_function not in ACTIVATION_FUNCTIONS:
            raise Exception('Função não suportada')
        self.activation_function_name = activation_function
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
        self.activation_function_grad = ACTIVATION_FUNCTIONS_GRADIENTS[activation_function]

    def train(self, X, Y):
        iter_num = 0
        e = 10
        while np.abs(e).mean() > self.min_loss_error and iter_num < self.max_iter_num:
            iter_num += 1

            h0 = self.predict(X)
            diff = (h0 - Y) * self.activation_function_grad(X.dot(self.weights) + self.bias) * X
            e = diff.mean()
            bias_e = (h0 - Y).mean()

            self.saved_errors.append(e.sum())
            self.weights = self.weights - self.learning_rate * e
            self.bias = self.bias - self.learning_rate * bias_e

            print(f"Época: {iter_num} Erro: {e}")
        return iter_num

    def predict(self, X):
        return self.activation_function(X.dot(self.weights) + self.bias)

    def predict_single(self, x):
        return self.activation_function(np.array(x).dot(self.weights) + self.bias)
