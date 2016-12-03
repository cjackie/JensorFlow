import numbers

class BackpropNode:
    def diveritive(self):
        raise Exception("BackpropNode: need to implement diveritive")

    def compute(self):
        raise Exception("BackpropNode: need to implement compute")

    def forward(self):
        raise Exception("BackpropNode: need to implement forward")

    def backward(self):
        raise Exception("BackpropNode: need to implement backward")


class PolynomialNode(BackpropNode):
    def __init__(self, coefs):
        '''
        @coefs :dictionary-like. 
            mapping degree of polynomial to its coeffients. for degrees
            not in the mapping, they are assigned 0 coeffients.
        '''
        if (not isinstance(coefs, dict)):
            raise TypeError("coef1")
        for degree, coef in coefs.items():
            if (not isinstance(degree, int) and not isinstance(coef, numbers.Number)):
                raise TypeError("coef2")

        self._coefs = dict(coefs)

    def compute(self):
        result = 0
        for degree, coef in coefs.items():
            