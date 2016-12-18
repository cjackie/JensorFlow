from typechecker import accept
from LayerBase import LayerInterface


class ConvolutionLayer(LayerInterface):


    def __init__(self, filterDimension, neuronNum, inputImageDimension):
        '''
        @filterDimension :tuple (height, width, depth).
        @neuronNum :int. determines number of outputs images.
        @inputImageDimension :tuple(height :int, width :int, depth:int).
        '''
        self.__filterDimension = filterDimension
        self.__neuronNum = neuronNum
        self.__inputImageDimension = inputImageDimension
        self.__outputDimension = inputImageDimension[0], inputImageDimension[1], neuronNum
        # initialize thetas and bias
        height, width, depth = filterDimension
        thetasDimension = height, width, depth, neuronNum
        self.__thetas = ndarray(shape=thetasDimension)
        self.__bias = ndarray(shape=(neuronNum))


    def _forwardAt(self, inputImage, iPrime, jPrime, fPrime):
        IInput, JInput, FInput = self.__inputImageDimension
        IFilter, JFilter, FFilter, _ = self.__thetas.shape
        bias = self.__bias[fPrime]
        r = 0.0
        for i in IFilter:
            for j in JFilter:
                for f in FFilter:
                    it = iPrime + i - 1
                    jt = jPrime + j - 1
                    ft = f
                    if (it < IInput and it >= 0) and (jt < JInput and jt >= 0) 
                        and (ft < FInput and ft >= 0):
                        r += inputImage[it, jt, ft] * self.__thetas[i, j, f, fPrime]
        return f

    def forward(self, inputImage):
        '''
        @inputImage :ndarray of shape(height, width, depth).
        '''
        assert isinstance(inputImage, ndarray)
        assert inputImage.shape == self.__inputImageDimension
        IPrime, JPrime, FPrime = self.__outputDimension
        outputImage = ndarray(shape=self.__outputDimension)
        for iPrime in IPrime:
            for jPrime in JPrime:
                for fPrime in FPrime:
                    outputImage[iPrime, jPrime, fPrime] = self._forwardAt(inputImage, iPrime, jPrime, fPrime)
        return outputImage

    def _gradientEvalAt(self, i, j, f, fPrime, inputImage, lossDerivUpperLayer):
        I, J, F = self.__inputImageDimension
        ILoss, JLoss, FLoss = lossDerivUpperLayer.shape
        r = 0.0
        for iPrime in range(ILoss):
            for jPrime in range(JLoss):
                ti = iPrime + i - 1
                tj = jPrime + j - 1
                if (ti < I and ti > 0) and (tj < J and tj > 0):
                    r += lossDerivUpperLayer[iPrime][jPrime][fPrime] * \
                        inputImage[ti][tj][f]
        return r


    def computeParamGradients(self, lossDerivUpperLayer, inputImage):
        '''
        @lossDerivUpperLayer :ndarray of shap tuple (height, width, depth).
            the shape is same as the output from @forward method..
        @inputImage :ndarray of shape(height, width, depth).
        @return :ndarray of shape @theta.
        '''
        assert lossDerivUpperLayer.shape == self.__outputDimension
        assert inputImage.shape == self.__inputImageDimension
        I, J, F, FPrime = self.__thetas.shape
        gradients = ndarray(shape=self.__thetas.shape)
        for i in range(I):
            for j in range(J):
                for f in range(F):
                    for fPrime in range(FPrime):
                        gradients[i,j,f,fPrime] = self._gradientEvalAt(i,j,f,fPrime,inputImage,lossDerivUpperLayer)
        return gradients

    def _backpropAt(self, i, j, f, lossDerivUpperLayer):
        IPrime, JPrime, FPrime = lossDerivUpperLayer.shape
        thethaDimen = self.__thetas.shape
        r = 0.0
        for iPrime in IPrime:
            for jPrime in JPrime:
                for fPrime in FPrime:
                    ti = i - iPrime + 1
                    tj = j - jPrime + 1
                    tf = f
                    tfPrime = fPrime
                    if (ti < thethaDimen[0] and ti >= 0) and (tj < thethaDimen[1] and tj >= 0)
                        and (tf < thethaDimen[2] and tf >= 0) and (tfPrime < thethaDimen[3] and tfPrime > 0):
                        r += lossDerivUpperLayer[iPrime, jPrime, fPrime] * self.__thetas[ti, tj, tf, tfPrime]
        return r

    def backprop(self, lossDerivUpperLayer):
        '''
        @lossDerivUpperLayer :ndarray. The shape is the outputImage shape.
        @return :ndarray. The shape is inputImage.
        '''
        I, J, F = self.__inputImageDimension
        lossDerivs = ndarray(shape=self.__inputImageDimension)
        for i in I:
            for j in J:
                for f in F:
                    lossDerivs[i,j,f] = self._backpropAt(i,j,f,lossDerivUpperLayer)
        return lossDerivs


    def getParams(self):
        return self.__thetas.copy()

    def setParams(self, theta):
        '''
        @theta :ndarray of same shape of self.__theta.
        '''
        assert theta.shape == self.__theta.shape
        self.__theta = theta.copy()


