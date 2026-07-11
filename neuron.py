import math
import numpy as np 
import matplotlib.pyplot as plt
import random
from main import Value, draw_dot


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        out = act.tanh()
        return out 
    
    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs 

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps =  neuron.parameters()
            params.extend(ps)
        return params
    


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin]+nouts 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



# x = [2.0,3.0, -1.0]

# o = n(x)
# print(n.parameters())
# dot = draw_dot(o)
# dot.render(directory='doctest-output', view=True)


if __name__ == "__main__":
    n = MLP(3, [4,4,1])
    xs = [
        [2.0,-1.0, 3.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]



    for k in range(1000):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout-ygt)**2 for yout, ygt in zip(ys, ypred))

        # zero grad
        for p in n.parameters():
            p.grad = 0.0

        # backward pass
        loss.backward()

        # gradient descent 

        for p in n.parameters():
            p.data += -0.05*p.grad

        if(k %10  == 0): print(k, loss.data)
    
    print(ypred)

