# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, 
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
	self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Write your code to train the perceptron here

	index = 0;
	for picture in self.trainingSet:
		#training the different pictures in the dataset
		#first classifie them
		if self.classify(picture):
			#classified as 7
	
			if self.trainingSet.label[index]  == 0:
				#but it is not a 7
				#error = prediction - expected = 1 - 0
				self.updateWeights(picture, 1)

		else:
			if self.trainingSet.label[index]  == 1:
				#but it is a 7
				#error = prediction - expected = 0 - 1
				self.updateWeights(picture, -1)

		index += 1
		

        pass


    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Write your code to do the classification on an input image
	
	#add the bias w0 to the sum
	ret = 1.0;
	index = 0;	
	for value in testInstance:
		ret += value *	self.weight[index]
		index += 1

	return ret > 0.0

        pass

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

    def updateWeights(self, input, error):
        # Write your code to update the weights of the perceptron here

	index=0
	while index < len(self.weight):
		#update every weight for every input
		self.weight[index] = self.weight[index]- self.learningRate * error * input[index]
		index += 1
	
        pass
         
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
	return Activation.sign(np.dot(np.array(input), self.weight))












