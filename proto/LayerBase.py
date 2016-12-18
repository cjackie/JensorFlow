from abc import *
from typechecker import accept
from numpy import matrix, ndarray

class LayerInterface():
    __metaclass__ = ABCMeta

    @abstractmethod
    @accept(object, matrix)
    def forward(self, inputVector):
        '''
        compute forward value.
        @inputVector :matrix of shape (1, inputVectorSize). input data
        '''
        pass

    @abstractmethod
    @accept(object, matrix)
    def backprop(self, lossDerivUpperLayer):
        '''
        compute accumulative loss derivative
        @lossDerivUpperLayer :matrix of shape (outputVectorSize, 1). 
            derivative wrt the loss from the upper layer.
        '''
        pass
    
    @abstractmethod   
    @accept(object, matrix, matrix)
    def computeParamGradients(self, lossDerivUpperLayer, inputVector):
        '''
        compute the gradient wrt the loss of each parameter.
        @return :matrix of shape (self._nInputs*self._nOutputs, 1).
        '''
        pass

    @abstractmethod
    @accept(object)
    def getParams(self):
        '''
        return a vector of parameters.
        @return :matrix of shape (self._nInputs*self._nOutputs, 1). 
        '''
        pass

    @abstractmethod
    @accept(object, matrix)
    def setParams(self, thetas):
        pass