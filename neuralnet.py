import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def inverse_sigmoid(x):
    return sigmoid(x) / (1.0 - sigmoid(x))
 
class NeuralNet():

    def __init__(self, size_params, training_data, debug=False):
        if not self._check_input_types(size_params, training_data):
            
            self.num_of_layers = len(size_params)
            self.weights = self.initialise_weights(size_params)
            self.biases = self.initialise_biases(size_params)
            
            self.activations = [0] * (self.num_of_layers)
            self.error = 0.0
            
            self.training_data = training_data

            if debug:
                print("Init weights: ", self.weights)
                print("Init biases: ", self.biases)

            self.biases = self.initialise_biases(size_params)

    def train(self):
        for idx in range(0, len(self.training_data[0])):
            self.activations[0] = np.array(self.training_data[0][idx])
            self.activations[0].shape = (2,1)
            
            self.feed_forward()
#            print(self.activations[-1])
            self.compute_error(self.activations[-1], np.array(self.training_data[-1][idx]).reshape(2,1))
#            print(self.error)
            
            self.back_propogate()
            
            
            

    def _check_input_types(self, size_params, data):
        error = False
        
        if not isinstance(size_params, list):
            print("size_params should be a list, not ", type(size_params))
            error = True

        if not isinstance(data, list):
            print("data should be a list, not ", type(data))
            error = True

        if len(data) != 2:
            print("data should have two elements (list of input data and a list of expected outputs), not ", len(data))
            error = True

        if not isinstance(data[0], list):
            print("first element of data should be a list of input data, not ", type(data[0]))
            error = True

        if not isinstance(data[1], list):
            print("second element of data should be a list of expected outputs, not ", type(data[0]))
            error = True

        for item in data[0]:
            if len(item) != size_params[0]:
                print("each element in input data should have length ", size_params[0], ", not ", len(item), " (", item, ")")
                error = True

        for item in data[1]:
            if len(item) != size_params[-1]:
                print("each element in expected outputs should have length ", size_params[-1], ", not ", len(item), " (", item, ")")
                error = True

        return error

    def initialise_weights(self, size_params):

        temp_weights = [0] * len(size_params)

        for layer in range(1, self.num_of_layers):
            temp_weights[layer] = np.random.random((size_params[layer], size_params[layer-1]))

        return temp_weights

    def initialise_biases(self, size_params):

        temp_biases = [0] * len(size_params)

        for layer in range(1, self.num_of_layers):
            temp_biases[layer] = np.random.random((size_params[layer], 1))

        return temp_biases
    
    def compute_activation(self, weights, biases, prev_layer):
        return sigmoid(np.dot(weights, prev_layer) + biases)
    
    def compute_error(self, calculated, expected):
        #self.error = ((expected - calculated)**2)*0.5
        
        delta = -(expected - calculated) * calculated * (1.0 - calculated)
        #print("Calculated: ", calculated)
        #print("Expected: ", expected)
        print("Delta: ", delta)
    
    def back_propogate(self):
        for layer in range(self.num_of_layers-1, 0, -1):
            print(inverse_sigmoid(self.activations[layer] - self.biases[layer]))
            #self.activations[layer-1]
    
    def feed_forward(self):
        
        for layer in range(1, self.num_of_layers):
#            print("Layer ", layer)
#            print("Weights: ", self.weights[layer])
#            print("Biases: ", self.biases[layer])
#            print("Prev Layer: ", self.activations[layer - 1])
            self.activations[layer] = self.compute_activation(self.weights[layer], self.biases[layer], self.activations[layer -1])
        

if __name__ == "__main__":


    size = [2, 3, 2]
    input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_data = [[0, 0], [0, 1], [0, 1], [1, 1]]

    data = [input_data, expected_data]

    nnet = NeuralNet(size, data, debug=True)
    nnet.train()