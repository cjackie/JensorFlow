'''
Linear Layer in a deep learning network. three key things:
- backprop
- forward
- drivative matrix of thetas(weights)
'''

from typechecker import accept
from numpy.matlib import rand, matrix
from numpy import ndarray

class LinearLayer():

    @accept(LinearLayer, int, int)
    def __init__(self, nInputs, nOutputs):
        self._nInputs = nInputs
        self._nOutputs = nOutputs
        # initialize parameters
        self._thetas = rand(nInputs, nOutputs)

    @accept(LinearLayer, matrix)
    def forward(self, inputVector):
        '''
        compute forward value.
        @inputVector :matrix of shape (1, @self._nIutputs). input data
        '''
        assert inputVector.shape == (1, self._nInputs)
        return inputVector * self._thetas

    @accept(LinearLayer, matrix)
    def backprop(self, lossDerivUpperLayer):
        '''
        compute accumulative loss derivative
        @lossDerivUpperLayer :matrix of shape (@self._nOutputs, 1). 
            derivative wrt the loss from the upper layer.
        '''
        assert lossDerivUpperLayer.shape == (self._nOutputs, 1)
        return self._thetas * lossDerivUpperLayer

    @accept(LinearLayer, matrix, matrix)
    def computeParamGradients(self, lossDerivUpperLayer, inputVector):
        '''
        compute the gradient wrt the loss of each parameter.
        @return :matrix of shape (self._nInputs*self._nOutputs).
        '''
        assert lossDerivUpperLayer.shape == (self._nOutputs, 1)
        assert inputVector.shape == (1, self._nInputs)
        gradients = lossDerivUpperLayer * inputVector
        return gradients.reshape(self._nInputs*self._nOutputs)

    def getParams(self):
        '''
        return a vector of parameters.
        @return :matrix of shape (self._nInputs*self._nOutputs, 1). 
        '''
        return self._thetas.reshape(self._nInputs*self._nOutputs, 1)

    @accept(LinearLayer, matrix)
    def setParams(self, thetas):
        '''
        set the parameters.
        @thetas :matrix of shape (self._nInputs*self._nOutputs, 1). 
        '''
        assert thetas.shape == (self._nIutputs*self._nOutputs, 1)
        self._thetas = thetas.copy()
        self._thetas.reshape(self._nInputs, self.nOutputs)


if '__main__' == __name__:
    layer = LinearLayer(3, 2)
    # parameters
    parameters = layer.getParams()
    message = "Parameters is " + str(parameters)
    print(message)
    # forward pass
    inputs = matrix([2,3,4])
    forwardValues = layer.forward(inputs)
    message = "With forward inputs " + str(inputs) + \
        ", forward values is " + str(forwardValues)
    print(message)
    # backprop
    accumulativeErrorVector = matrix([[2], [3]])
    backpropValues = layer.backprop(accumulativeErrorVector)
    message = "Backprop output is " + str(backpropValues)
    print(message)
    # get gradients
    gradients = gradients(accumulativeErrorVector, inputs)
    message = "Gradient is " + str(gradients)
    print(message)







