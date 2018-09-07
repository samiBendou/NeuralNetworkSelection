import numpy as np

SIZE_IN = 3;
SIZE_OUT = 3;
SIZE_INT = 2;

def sigmoid(x, derivative=False):
  return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

class NeuronLayer :
    def __init__(this, size, activation = sigmoid):        
        this.set(size, activation): 

    def activate(this):
        this.activation(this.vector);

    def set(this, size, activation = sigmoid): 
        this.size = size;
        this.activation = activation;
        this.vector = np.zeros(size);
   
   def reinit(this):
       this.vector = np.zeros(size);

class NeuralNetwork :
    def __init__(this, shape, activation = sigmoid, values = None):
        this.coefs = [];
        this.layers = [];
        if values == None:
            this.values = [];
            for i in range(len(shape) - 1):
                this.values = this.values + [0.0] * shape[i] * shape[i + 1];
        this.values = np.array(this.values);
        this.putValuesInCoefs();
        for i in range(len(shape)):
            this.layers.append(NeuronLayer(shape[i], activation)); 

    def putValuesInCoefs(this, values = None):
        if values != None:
            this.values = values;
        this.coefs = []
        j = 0;
        for i in range(len(this.shape) - 1):
            matrix = np.zeros(this.shape[i], this.shape[i + 1]);
            for x in range(this.shape[i]):
                for y in range(this.shape[i]):
                    matrix[x,y] = this.values[j];
                    j++;
            this.coefs.append(matrix);
    
    def feedInput(this, values):
        if (len(values) == len(this.layers[0].size)):
            for i in range(len(values)):
                this.layers[0].vector[i] = values[i];
            
"""
class NeuralNetwork :
    def __init__(this, weightMatrix = None):
        this.vectorIn = np.mat(1, SIZE_IN + 1);

        this.vectorOut = np.mat(1, SIZE_OUT);

        this.neuralMatrix = [np.mat(1, SIZE_INT + 1), np.mat(1, SIZE_INT + 1)];

        if weightMatrix == None :

            this.weightMatrix = [np.random.rand(SIZE_IN + 1, SIZE_INT + 1),
                                 np.random.rand(SIZE_INT + 1, SIZE_INT + 1),
                                 np.random.rand(SIZE_INT + 1, SIZE_OUT + 1)];


    def computeOutput(this, vectorIn):
        sizeOfNetwork = len(this.neuralMatrix);

        if len(vectorIn) == SIZE_IN + 1:
            vectorIn[-1] = 1;
            this.neuralMatrix[0] = this.weightMatrix[0] * vectorIn
            this.neuralMatrix[0][-1] = 1
            for k in range(1, sizeOfNetwork):
                this.neuralMatrix[k] = this.weightMatrix[k] * this.neuralMatrix[k - 1]
                this.neuralMatrix[k] = np.matrix

            this.vectorOut = this.weightMatrix[-1] * this.neuralMatrix[-1]
"""
