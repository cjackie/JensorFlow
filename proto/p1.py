# basic interface a node has
class NodeBase:
    def getOutput(self):
        raise Exception("need to implement getOutput")

    def getState(self):
        raise Exception("need to implement getState")


class Add(NodeBase):
    def __init__(self, node1, node2):
        '''
        @node1: NodeBase
        @node2: NodeBase
        '''
        self.node1 = node1
        self.node2 = node2
        self.state = None

    def getOutput(self):
        self.state = self.node1.getOutput() + self.node2.getOutput()
        return self.state

    def getState(self):
        return self.state


class Constant(NodeBase):
    def __init__(self, num):
        '''
        @num: number
        '''
        self.num = num

    def getOutput(self):
        return self.num

    def getState(self):
        return self.num

def run(node):
    node.getOutput();


if __name__ == '__main__':
    '''
    following dataflow:

    one ------|
               -->  add1 --------|
               -->               |
    two ------|                  |
                                 ----->  add2 -----> 
    three ---------------------------->  
    '''                
    one = Constant(1)
    two = Constant(2)
    three = Constant(3)
    add1 = Add(one, two)
    add2 = Add(add1, three)
    run(add2)
    print("result is " + str(add2.getState()))
    assert add2.getState() == 6

