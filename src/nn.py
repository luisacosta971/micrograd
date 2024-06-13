import random
from value import Value

class Neuron:
    
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, input):
        act = sum((wi * xi for wi, xi in zip(self.w, input)), self.b)
        out = act.tanh()
        return out
    
class Layer:
    
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, input):
        outs = [n(input) for n in self.neurons]
        return outs
    
class MLP:
    
    def __init__(self):
        pass
            
X = [1, 2, 3]
n = Neuron(3)
print(n(X))