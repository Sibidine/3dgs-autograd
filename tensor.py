import math
import numpy as np 
import matplotlib.pyplot as plt
from graphviz import Digraph 


        
def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape): # collapse size-1 axes that were stretched
        if s == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:

    def __init__(self,data, _children=(), _op='', label=''):
        self.data = np.asarray(data, dtype=np.float64) 
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None 
        self._prev = set(_children)
        self._op = _op
        self.label=label

    
    
    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data+other.data, (self,other), '+')
        def backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = backward
        return out

    def __rmul__(self, other):
        return self*other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data*other.data, (self,other),'*')
        def backward():
            self.grad += unbroadcast(other.data*out.grad, self.data.shape)
            other.grad += unbroadcast(self.data*out.grad, other.data.shape)
        out._backward = backward
        return out


    
    def __neg__(self):
        return self*-1
    
    def __sub__(self, other):
        return self+(-other)

    def __radd__(self, other):
        # float + Tensor -> becomes Tensor + float
        return self + other

    def __rsub__(self, other):
        # float - Tensor -> becomes float + (-Tensor)
        # (This will automatically trigger __radd__ under the hood!)
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def  __rtruediv__(self, other):
        return (self** -1)*other
    
    def __pow__(self, other):
        assert isinstance(other ,(int, float))
        out  = Tensor(self.data**other, (self, ), f'**{other}')

        def backward():
            self.grad += unbroadcast((other*self.data**(other-1))*out.grad, self.data.shape)
        out._backward = backward
        return out


    def tanh(self):
        x = self.data 
        t = (np.exp(2*x)-1)/(np.exp(2*x)+1)
        out = Tensor(t, (self, ), 'tanh')
        def backward():
            self.grad += (1- t**2)*out.grad
        out._backward = backward
        return out

    def sigmoid(self):      
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self, ), 'sigmoid')
        def backward():
            self.grad += s*(1-s)*out.grad
        out._backward = backward
        return out

    def sqrt(self):
        out = self**0.5
        # def backward():
        #     self.grad += 0.5*self**(-0.5)*out.grad 
        # out._backward = backward 
        return out 

    def relu(self):
        out = Tensor(np.maximum(0,self.data), (self,), 'relu')
        def backward():
             self.grad += (out.data > 0) * out.grad
        out._backward = backward
        return out

    def __abs__(self):
        out = Tensor(np.abs(self.data), (self,), 'abs')
        def backward(): 
            self.grad += np.where(self.data >= 0, 1.0, -1.0) * out.grad
        out._backward = backward
        return out

    def exp(self):
        x = self.data 
        out = Tensor(np.exp(x), (self, ), 'exp') 

        def backward():
            self.grad +=  out.data * out.grad
        out._backward = backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        def bw():
            self.grad  += unbroadcast(out.grad @ other.data.swapaxes(-1,-2), self.data.shape)
            other.grad += unbroadcast(self.data.swapaxes(-1,-2) @ out.grad, other.data.shape)
        out._backward = bw
        return out

    def swapaxes(self, a, b):
        out = Tensor(self.data.swapaxes(a, b), (self,), 'swap')
        def bw():
            self.grad += out.grad.swapaxes(a, b)   # swap the same axes back
        out._backward = bw
        return out

    def stack(tensors, axis=0):                    # standalone fn: takes a list
        out = Tensor(np.stack([t.data for t in tensors], axis=axis), tuple(tensors), 'stack')
        def bw():
            pieces = np.moveaxis(out.grad, axis, 0)   # bring stack axis to front -> (n, ...)
            for t, g in zip(tensors, pieces):
                t.grad += g                           # hand slice i back to input i
        out._backward = bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)          # restore the reduced axis
            self.grad += np.ones_like(self.data) * g # then broadcast back up
        out._backward = bw
        return out

    def __getitem__(self, idx):                      # this is how model.means[vis] stays differentiable
        out = Tensor(self.data[idx], (self,), 'getitem')
        def bw():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)              # scatter-add handles masks & repeated indices
            self.grad += g
        out._backward = bw
        return out
        
        
    def log(self):
        out = Tensor(np.log(self.data), (self, ), 'log')
        def backward():
            self.grad += 1/self.data * out.grad 
        out._backward = backward
        return out 

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self, ), "reshape")
        def backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = backward
        return out 
    
    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)
        count = self.data.size/s.data.size 
        return s*(1.0/count)
    
    def clamp(self, min=None, max=None):
        lo = -np.inf if min is None else min 
        hi = np.inf if max is None else max 
        out = Tensor(np.clip(self.data, lo, hi), (self, ), "clamp")
        def backward():
            mask = (self.data >= lo) & (self.data <= hi)
            self.grad += mask*out.grad 
        out._backward = backward 
        return out


    def backward(self):
        # self.grad = np.ones_like(self.data)
        # topo = []
        # visited = set()
        # def build_topo(v):
        #     if v not in visited:
        #         visited.add(v)
        #         for child in v._prev:
        #             build_topo(child)
        #         topo.append(v)
        
        # self.grad = 1
        # build_topo(self)
        # for node in reversed(topo):
        #     node._backward()

        topo, visited  =[], set()
        stack = [(self, False)]
        while stack:
            v, processed = stack.pop()
            if processed:
                topo.append(v)
                continue 
            if id(v) in visited:
                continue 
            visited.add(id(v))
            stack.append((v, True))
            for child in v._prev:
                stack.append((child, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()



class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps 
        self.t = 0 
        self.m = [np.zeros_like(p.data) for p in self.params] #1st momentum 
        self.v = [np.zeros_like(p.data) for p in self.params] #2nd

    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1-self.b1)*g 
            self.v[i] = self.b2 * self.v[i] + (1-self.b2)*g*g
            m_hat = self.m[i] / (1-self.b1**self.t)
            v_hat = self.v[i] / (1-self.b2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) * self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data) 




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
        # for any Tensor in the graph, create a record for it
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if Tensor is the result fo some op, create node for it
            dot.node(name=uid+n._op, label=n._op)
            dot.edge(uid+n._op,uid)

        
    for n1,n2  in edges:
        dot.edge(str(id(n1)), str(id(n2))+n2._op)
    
    return dot




def vdot(a, b):
    return sum((ai*bi for ai, bi in zip(a, b)), Tensor(0.0))

def matvec(M, v):
   return [vdot(row, v) for row in M]

def matmul(A, B):
    Bt = list(zip(*B))
    return [[vdot(r, c) for c in Bt] for r in A]

def transpose(M):
    return [list(r) for r in zip(*M)]

# test2()

