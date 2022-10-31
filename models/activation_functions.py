import numpy as np 

ACTIVATION_FUNCTIONS = {
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1/(1 + np.exp(-x)),
    'identity': lambda x: x,
}

ACTIVATION_FUNCTIONS_GRADIENTS = {
    'relu': lambda x: np.where(x > 0, 1, 0),
    'sigmoid': lambda x: ACTIVATION_FUNCTIONS['sigmoid'](x) * (1 - ACTIVATION_FUNCTIONS['sigmoid'](x)),
    'identity': lambda x: 1,
}
