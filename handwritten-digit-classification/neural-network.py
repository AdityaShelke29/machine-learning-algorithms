import random
import numpy as np

# Creating a Neural Nework to Classify Handwritten Digits

class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.weights = []

        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i], layers[i - 1]))
        
        self.biases = []

        for i in range(1, len(layers)):
            self.biases.append(np.random.randn(layers[i]))
    
    def forwardPropagation(self, input):
        activation = input

        for weight, bias in zip(self.weights, self.biases):
            
            activation = self.sigmoid(np.dot(weight, activation) + bias)
        
        return activation
    
    def sigmoid(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    # Functions for Datalogging and Debugging

    # The following 4 functions are utilized in dataloggin and debugging. 
    # They can be used to print out the structure of the neural network, 
    # and display each of the weight matrices, the bias column vectors, 
    # and the activation column vectors.

    # def printStructure(self)
    # -------------------------------------------------------------------------
    def printStructure(self):
        print("\n\n")
        print("Number of Layers: " + str(self.num_layers) + "\n")

        for i in range(self.num_layers - 1):
            print("Weights Between Layer " + str(i) + 
                  " and Layer " + str(i + 1) + ":\n")
            print(self.returnMatrix(self.weights[i]))
        
        for i in range(1, self.num_layers):
            print("Biases for Layer " + str(i) + ":\n")
            print(self.returnVector(self.biases[i - 1]))
    # -------------------------------------------------------------------------

    # def returnMatrix(self, matrix)
    # -------------------------------------------------------------------------
    def returnMatrix(self, matrix):
        output = ""

        for row in matrix:
            for column in row:
                output = output + str(self.truncate_float(column, 2)) + "   "
            output = output + "\n"
        
        return output
    # -------------------------------------------------------------------------
    
    # def returnVector(self, vector)
    # -------------------------------------------------------------------------
    def returnVector(self, vector):
        output = ""

        for row in vector:
            output = output + str(self.truncate_float(row, 2)) + "\n"
        
        return output
    # -------------------------------------------------------------------------

    # def truncate_float(self, number)
    # -------------------------------------------------------------------------
    def truncate_float(self, float_number, decimal_places):
        multiplier = 10 ** decimal_places
        return int(float_number * multiplier) / multiplier
    
    # -------------------------------------------------------------------------

# Main Method
network = NeuralNetwork([3, 4, 2])

network.printStructure()
network.forwardPropagation([1, 2, 3])