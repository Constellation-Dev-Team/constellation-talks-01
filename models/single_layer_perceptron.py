import numpy as np
import pandas as pd

class SingleLayerPerceptron:
    weights = []
    learning_rate = 0.01
    
    def __init__(self, size):
        self.n_features = size
        self.weights = np.ones(size) * 1/size
        self.saved_errors = []
        self.bias = 0
    
    def train(self, X, Y):
        iter_num = 0
        before_weight = self.weights + 10
        # try:
        while np.mean(np.abs(before_weight - self.weights)) > 1e-8 and iter_num < 100000:
            before_weight = self.weights
            iter_num += 1

            h0 = self.predict(X)
            diff = (h0 - Y)
            e =  diff.mean()* self.weights
            bias_e = diff.mean()
                
            self.saved_errors.append(e.max())
            self.weights = self.weights - self.learning_rate * e
            self.bias = self.bias - self.learning_rate * bias_e
        # except:
        #     print('Erro no treinamento dos dados')
        return iter_num
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

        
    def predict_single(self, x):
        return np.array(x).dot(self.weights) + self.bias