# basic interface a node has
class NodeBase:
    def getOutput(self):
        raise Exception("NodeBase: need to implement getOutput")

    def getState(self):
        raise Exception("NodeBase: need to implement getState")


class VariableBase:
    def setVariable(self):
        raise Exception("VariableBase: need to implement setVariable")

    def getVariable(self):
        raise Exception("VariableBase: need to implement getVariable")


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


class Variable(VariableBase, NodeBase):
    def __init__(self, val, name):
        self.val = val
        self.name = name

    def setVariable(self, val):
        self.val = val

    def getVariable(self):
        return self.val

    def getOutput(self):
        return self.val

    def getState(self):
        return self.val


class Assign(NodeBase):
    def __init__(self, var, node):
        '''
        @var: VariableBase
        @node: NodeBase
        '''
        self.var = var
        self.node = node
        self.state = None

    def getOutput(self):
        self.var.setVariable(self.node.getOutput())
        self.state = self.node.getState()
        return self.state

    def getState(self):
        return self.state


def run(node):
    node.getOutput();




if __name__ == '__main__':
    def demo1():
        '''
        simply adding three numbers. dataflow diagram:
        one ------|
                   -->  add1 -------|
                   -->              |
        two ------|                 |
                                    |----->  add2 -----> 
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

    demo1()

    def demo2():
        '''
        a simple counter. dataflow diagram:
            ......................................|
            |                                     |       
            v                                     |
        counter --------->  add  -----> assign  ...
                     |---> 
                     |
        one ---------|

        in there "assign" is kind of wierd...
        '''
        counter = Variable(0, "counter")
        one = Constant(1)
        add = Add(counter, one)
        assign = Assign(counter, add)
        for i in range(5):
            run(assign)
            print("demo2: output is " + str(assign.getState()))
        assert counter.getVariable() == 5

    demo2()
