import math
import numpy as np 
import matplotlib.pyplot as plt
from graphviz import Digraph 

class Value:

    def __init__(self,data, _children=(), _op='', label=''):
        self.data = data 
        self.grad = 0.0
        self._backward = lambda: None 
        self._prev = set(_children)
        self._op = _op
        self.label=label

    
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self,other), '+')
        def backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = backward
        return out

    def __rmul__(self, other):
        return self*other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self,other),'*')
        def backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = backward
        return out
    
    def __neg__(self):
        return self*-1
    
    def __sub__(self, other):
        return self+(-other)

    def __radd__(self, other):
        # float + Value -> becomes Value + float
        return self + other

    def __rsub__(self, other):
        # float - Value -> becomes float + (-Value)
        # (This will automatically trigger __radd__ under the hood!)
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other ,(int, float))
        out  = Value(self.data**other, (self, ), f'**{other}')

        def backward():
            self.grad += (other*self.data**(other-1))*out.grad
        out._backward = backward
        return out


    def tanh(self):
        x = self.data 
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')
        def backward():
            self.grad += (1- t**2)*out.grad
        out._backward = backward
        return out

    def sigmoid(self):      
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self, ), 'sigmoid')
        def backward():
            self.grad += s*(1-s)*out.grad
        out._backward = backward
        return out

    def sqrt(self):
        out = self**0.5
        def backward():
            self.grad = 0.5*self**(-0.5)*out.grad 
        out._backward = backward 
        return out 

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')
        def backward():
             self.grad += (out.data > 0) * out.grad
        out._backward = backward
        return out

    def __abs__(self):
        out = Value(abs(self.data), (self,), 'abs')
        def backward(): 
            self.grad += (1.0 if self.data >= 0 else -1.0) * out.grad
        out._backward = backward
        return out

    def exp(self):
        x = self.data 
        out = Value(math.exp(x), (self, ), 'exp') 

        def backward():
            self.grad +=  out.data * out.grad
        out._backward = backward
        return out

    def backward(self):
        self.grad = 1
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        self.grad = 1
        build_topo(self)
        for node in reversed(topo):
            node._backward()
        


def trace(root):
    nodes, edges = set(), set() 
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes, edges 


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'}) # left to right

    nodes, edges = trace(root) 
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a record for it
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if value is the result fo some op, create node for it
            dot.node(name=uid+n._op, label=n._op)
            dot.edge(uid+n._op,uid)

        
    for n1,n2  in edges:
        dot.edge(str(id(n1)), str(id(n2))+n2._op)
    
    return dot


def test2():
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    b = Value(8.0, label='b')

    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2 +b; n.label='n'
    e = (2*n).exp()
    o = ( e - 1)/(e + 1)
    # o = n.tanh(); o.label = 'o'
    o.backward()
    # top = topo(o) 

    # print(topo(o))


    dot = draw_dot(o)
    dot.render(directory='doctest-output', view=True)


def vdot(a, b):
    return sum((ai*bi for ai, bi in zip(a, b)), Value(0.0))

def matvec(M, v):
   return [vdot(row, v) for row in M]

def matmul(A, B):
    Bt = list(zip(*B))
    return [[vdot(r, c) for c in Bt] for r in A]

def transpose(M):
    return [list(r) for r in zip(*M)]

# test2()










