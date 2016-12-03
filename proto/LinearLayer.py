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

    @accept(LinearLayer, matrix)
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
        return self._thetas.reshape(self._nInputs*self._nOutputs)

    @accept(LinearLayer, ndarray)
    def setParams(self, thetas):
        '''
        set the parameters.
        @thetas :ndarray. 
        '''
        assert thetas.size == self._nIutputs*self._nOutputs
        self._thetas = matrix(thetas, copy=True)
        self._thetas.reshape(self._nInputs, self.nOutputs)

