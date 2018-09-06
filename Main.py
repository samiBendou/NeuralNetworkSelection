import numpy as np

SIZE_IN = 3;
SIZE_OUT = 3;
SIZE_INT = 2;

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

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

