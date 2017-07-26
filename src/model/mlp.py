
import numpy as np

from util.loss_functions import CrossEntropyError
from util.loss_functions import BinaryCrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='ce', learningRate=0.01, epochs=50, cost=0.5):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'ce':
            self.loss = CrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers # USELESS?!

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        #prevSize = 128
        #nextSize = 128
        # for i in (1:2):
        #     nextSize = prevSize-10
        #     self.layers.append(LogisticLayer(prevSize, nextSize, 
        #                    None, inputActivation, False)) # CHECK INPUT PARAMETERS
        #     prevSize = nextSize

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128-1, 10, 
                           None, outputActivation, True))

        self.inputWeights = inputWeights # USELESS?!

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """

        lastOutput = inp
        for i in range(0,len(self.layers)):
            lastOutput = self.layers[i].forward(lastOutput)

        return lastOutput # final softmax classification values
        
    def _compute_error(self, target, output):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the error
        """

        errorVector = self.loss.calculateError(target,output)
        
        return errorVector

    
    def _update_weights(self, learningRate, errorVector):
        """
        Update the weights of the layers by propagating back the error
        """
        errorVector = errorVector
        outputLayer = self._get_output_layer()
        outputLayer.computeDerivative(errorVector,np.ones(outputLayer.nOut))
        outputLayer.updateWeights(learningRate)

        nextDerivatives = outputLayer.deltas
        nextWeights = outputLayer.weights

        for i in range(2,len(self.layers)+1):
            layer = self.layers[-i]
            layer.computeDerivative(nextDerivatives,nextWeights)
            layer.updateWeights(learningRate)
            nextDerivatives = layer.deltas
            nextWeights = layer.weights
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for j in range(0,self.epochs):
            for i in range(0,len(self.trainingSet.input)):
                sampleIn = self.trainingSet.input[i]
                sampleLabel = self.trainingSet.label[i]
                sampleLabelVector = np.zeros(10)
                sampleLabelVector[sampleLabel] = 1
                output = self._feed_forward(sampleIn)
                errorVector = self._compute_error(sampleLabelVector, output)
                self._update_weights(self.learningRate,errorVector)

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here

        fireResult = self._feed_forward(test_instance)

        onehotPosition = np.where(fireResult==max(fireResult))[0][0]

        return onehotPosition


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
