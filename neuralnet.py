import numpy as np

class NeuralNet():
    
    def __init__(self, size_params, data, debug=False):
        if not self._check_input_types(size_params, data):
            self.num_of_hidden_layers = len(size_params) - 2
            self.weights = self.initialise_weights(size_params)
            self.biases = self.initialise_biases(size_params)
            
            if debug:
                print(self.weights)
                for weight in self.weights:
                    print(weight)
                print(self.biases)
                for bias in self.biases:
                    print(bias)
            
            self.biases = self.initialise_biases(size_params)
    
    def _check_input_types(self,size_params, data):
        if not isinstance(size_params, list):
            print ("size_params should be a list, not ", type(size_params))
            return True
        
        if not isinstance(data, list):
            print ("data should be a list, not ", type(data))
            return True
        
        if len(data) != 2:
            print("data should have two elements (list of input data and a list of expected outputs), not ", len(data))
            return True
        
        if not isinstance(data[0], list):
            print ("first element of data should be a list of input data, not ", type(data[0]))
            return True
        
        if not isinstance(data[1], list):
            print ("second element of data should be a list of expected outputs, not ", type(data[0]))
            return True
        
        for item in data[0]:
            if len(item) != size_params[0]:
                print ("each element in input data should have length ", size_params[0], ", not ", len(item), " (", item, ")")
                return True
        
        for item in data[1]:
            if len(item) != size_params[-1]:
                print ("each element in expected outputs should have length ", size_params[-1], ", not ", len(item), " (", item, ")")
                return True
        
        
        
        return False
    
    def initialise_weights(self,size_params):
        
        temp_weights = []
        
        for layer in range(1, self.num_of_hidden_layers + 2):
            temp_weights.append(np.random.random((size_params[layer], size_params[layer-1])))
        
        return temp_weights
    
    def initialise_biases(self, size_params):
        
        temp_biases = []
        
        for layer in range(1, self.num_of_hidden_layers + 2):
            temp_biases.append(np.random.random((size_params[layer],1)))
        
        return temp_biases

if __name__=="__main__":
    
    
    size_params = [2, 3, 2]
    input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_data = [[0, 0], [0, 1], [0, 1], [1, 1]]
    
    data = [input_data, expected_data]
    
    nnet = NeuralNet(size_params, data, debug=True)