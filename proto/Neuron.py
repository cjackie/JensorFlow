

class NeuroBase():
    '''
    call setup before use
    '''

    def __init__(self):
        '''
        @self._uid :number. unique identifer.
        @self._outNum :number. result after activation.
        @self._weights :list-like number. weights of inputs.
        '''
        self._uid = NeuroBase._yieldUid()
        self._outNum = None

    def _initializeWeights(self, inD):
        '''
        weight and inputs is associated by index. for example self._weights[2]
        is for self._inNums[2]
        @inD: int. number of inputs.
        '''
        self._weights = [0 for _i in range(inD)]

    def setup(self, inputNeuros, outputNeuros):
        '''
        connect with other neuros
        @inputNeuros :list-like NeuroBase. act as inputs.
        @nextNeuros :list-like NeuroBase. act as outputs.
        '''
        self._inputNeuros = inputNeuros[:]
        self._outputNeuros = outputNeuros[:]
        self._initializeWeights(len(inputNeuros))

    def activate(self):
        '''
        inputs can be obtained from calling _getOutNum on inputNeuros
        result from the neuro activition is stored in @self._outNum.
        '''
        raise Exception("NeuroBase: activate needs implemention")

    def _getOutNum(self):
        return self._outNum

    def _getUid(self):
        return self._uid

    def _getInputsAndWeights(self):
        '''
        helper function for getting inputs and weight.
        @return :tuple of (inputs :list-like number, weights :list-like number)
        '''
        inputs = [neuro._getOutNum() for neuro in self._inputNeuros]
        weights = self._weights[:]
        return (inputs, weights)


    # def equals(self, neuro):
    #     '''
    #     test if self is equal to @neuro by uid.
    #     @neuro :NeuroBase.
    #     @return :bool. 
    #     '''
    #     if self._uid == neuro._uid:
    #         return True
    #     else:
    #         return False


    NeuroNumber = '0'
    @staticmethod
    def _yieldUid():
        # WARNING, need to deal with overflow.
        newUid = int(NeuroBase.NeuroNumber) + 1
        NeuroBase.NeuroNumber = str(newUid)
        return NeuroBase.NeuroNumber

class ConstantNeuro(NeuroBase):
    '''
    no inputs to this neuro.
    '''
    def __init__(self, constant):
        NeuroBase.__init__(self)
        self._constant = constant

    def setup(self, outputNeuros):
        NeuroBase.setup(self, [], outputNeuros)

    def activate(self):
        self._outNum = self._constant


class LNeuro(NeuroBase):
    '''
    linear addition funciton, like w_0*x_0 + w_1*x_1 + w_2*x_2
    '''
    def activate(self):
        inputs, weights = self._getInputsAndWeights()
        assert len(inputs) == len(weights)
        n = 0.0
        for i in range(len(inputs)):
            n += inputs[i]*weights[i]
        self._outNum = n


if __name__ == '__main__':
    def testConstantNeuro():
        one = ConstantNeuro(1)
        one.setup([])
        assert one._getOutNum() == None
        one.activate()
        assert one._getOutNum() == 1
        one.activate()
        one.activate()
        assert one._getOutNum() == 1
        print("testConstantNeuro passed")
    testConstantNeuro()

    def testLNeuro():
        one = ConstantNeuro(1)
        two = ConstantNeuro(2)
        linearNeuro = LNeuro()
        one.setup([linearNeuro])
        two.setup([linearNeuro])
        linearNeuro.setup([one, two], [])
        one.activate()
        two.activate()
        linearNeuro.activate()
        assert linearNeuro._getOutNum() == 0
        linearNeuro._weights = [2.9, 3.2]
        linearNeuro.activate()
        assert linearNeuro._getOutNum() == 2.9*1+3.2*2
        print("testLNeuro passed")
    testLNeuro()


