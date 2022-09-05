import numpy as np

"""
Function definitions
"""

def activation_fn(input):
    #input: float
    return 1 if input >= 0 else 0

def perceptron(x,w,b):
    #x: list; w: list; b: float
    return activation_fn(np.dot(x,w) + b)

def NOT_perceptron(x):
    #x[i]: 0, 1
    return perceptron(x,w=-1,b=0)

def AND_perceptron(x):
    #x: [0,0], [0,1], [1,0], [1,1]
    return perceptron(x,w=[2,2],b=-4)

def OR_perceptron(x):
    #x: [0,0], [0,1], [1,0], [1,1]
    return perceptron(x,w=[1,1],b=-0.5)

def XOR_net(x): 
    layer_1_out = AND_perceptron(x)
    layer_2_out = [NOT_perceptron(layer_1_out), OR_perceptron(x)]
    return AND_perceptron(layer_2_out)
    
"""
Driver program
"""
x = [[0,0], [0,1], [1,0], [1,1]]

for i in x:
    print(XOR_net(i))