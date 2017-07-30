# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp
from numpy import divide
from numpy import ones
from numpy import asarray
import numpy as np



class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        # use e^x from numpy to avoid overflow
        return 1/(1+exp(-1.0*netOutput))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        return netOutput * (1.0 - netOutput)

    @staticmethod
    def tanh(netOutput):
        # return 2*Activation.sigmoid(2*netOutput)-1
        ex = exp(1.0*netOutput)
        exn = exp(-1.0*netOutput)
        return divide(ex-exn, ex+exn)  # element-wise division

    @staticmethod
    def tanhPrime(netOutput):
        # Here you have to code the derivative of tanh function
        return (1-Activation.tanh(netOutput)**2)

    @staticmethod
    def rectified(netOutput):
        return asarray([max(0.0, i) for i in netOutput])

    @staticmethod
    def rectifiedPrime(netOutput):
        # reluPrime=1 if netOutput > 0 otherwise 0
        #print(type(netOutput))
        return netOutput>0

    @staticmethod
    def identity(netOutput):
        return netOutput

    @staticmethod
    def identityPrime(netOutput):
        # identityPrime = 1
        return ones(netOutput.size)

    @staticmethod
    def softmax(netOutput):
        # Here you have to code the softmax function
        #Annahme netOutput ist ein Array mit den Werten
        #e = np.exp(netOutput)
        #eSum = np.sum(e)

        # The values are shifted by the max to get a better numerical stability
        shift = netOutput - np.max(netOutput)
        e = np.exp(shift)
        return e / np.sum(e)

        
    @staticmethod
    def softmaxPrime(netOutput):

        # Here you have to code the softmax function
        phi = Activation.softmax(netOutput)
        return phi * (1 - phi)

        # Tried to derive the jacoby matrix but didn't work out
        '''
        jacobyMatrix = np.zeros([len(netOutput), len(netOutput)])
        phi = Activation.softmax(netOutput)

        for i in range(0,len(netOutput)):
            for j in range(0, len(netOutput)):
                kroneckerDelta = 1 if (i == j) else 0
                jacobyMatrix[i, j] = phi[i] * (kroneckerDelta - phi[j])

        #res = np.dot(phi, jacobyMatrix)
        res = np.sum(jacobyMatrix, axis=1)
        #print(res)

        return res
        '''
        
    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'softmax':
            return Activation.softmaxPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
