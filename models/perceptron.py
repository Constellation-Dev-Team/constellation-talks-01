class Perceptron:
    weight = 0
    bias = 0
    learning_rate = 0.01
    
    def train(self, X, Y):
        iter_num = 0
        before_weight = self.weight + 10
        while abs(before_weight - self.weight) > 1e-9:
            e = 0
            e_bias = 0
            before_weight = self.weight
            iter_num += 1
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                h0 = x * self.weight + self.bias
                e += (h0 - y) * self.weight/len(X)
                e_bias += (h0 - y)/len(X)
            self.weight -= self.learning_rate * e
            self.bias -= self.learning_rate * e_bias
#             print(self.bias)
        return iter_num
    
    def predict(self, x):
        return x * self.weight + self.bias