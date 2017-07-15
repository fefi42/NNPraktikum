
import time

import numpy as np

from util.activation_functions import Activation
from util.loss_functions import BinaryCrossEntropyError
from util.loss_functions import DifferentError


class LogisticLayer():
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='sigmoid', isClassifierLayer=True):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.delta = np.zeros((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1))-0.5
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, input):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        #append bias for input
        self.input = np.append(input, [1])

        weightedinput = np.dot(self.weights, np.transpose(self.input))
        self.output =  Activation.sigmoid(weightedinput)

        return self.output

    def computeDerivative(self, nextDerivatives, nextWeights, label):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer

        This is backpropagation
        """


        if self.isClassifierLayer == True:
            de = DifferentError()
            bce = BinaryCrossEntropyError()

            #error = bce.calculateError(label, self.output)

            error = de.calculateError(label, self.output)
        else:
            error = np.dot(nextDerivatives, nextWeights)


        self.delta = Activation.sigmoidPrime(self.output) * error
        return self.delta # there should be one value in this vector



    def updateWeights(self, learningRate=0.1):
        """
        Update the weights of the layer
        """

        if self.isClassifierLayer == True:


            self.weights = self.weights + (learningRate * self.delta * self.input)

        else:
            self.weights = self.weights + learningRate * self.delta * self.input


        return self.weights


