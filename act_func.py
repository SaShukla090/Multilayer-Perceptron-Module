import numpy as np
class optimizer:
    def __init__(self):
       
        
        raise NotImplementedError
        
class activation_function:
    def __init__(self):
        self.name = None
        self.output = None
#         raise NotImplementedError
    def tanh(self, input_data):
        return np.tanh(input_data)
    
    def tanh_prime(self, input_data):
        return 1-np.tanh(input_data)**2
    def Relu(self, x):
        self.output = np.maximum(0,x)
        return self.output
    def Relu_prime(self, x):
#         self.out_prime = 
        self.output[self.output>0] = 1
        return self.output
    
    # softmax
    def softmax(self, x):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        s=x_exp/x_sum
        return s
        
        pass
    def softmax_prime(self, x):
        
        pass
    def activation(self, name, x):
        self.name = name
        if self.name == "relu":
            return self.Relu(x)
        elif self.name == "tanh":
            return self.tanh(x)
    
    def activation_prime(self,x):
#         self.name = name
        if self.name == "relu":
            return self.Relu_prime(x)
        elif self.name == "tanh":
            return self.tanh_prime(x)
     
           
    
        
        

       
    
        