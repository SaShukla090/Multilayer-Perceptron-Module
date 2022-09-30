import numpy as np
class loss_function:
    def __init__(self, name):
        self.name = name
        self.Y = None
        self.Yhat = None
#         raise NotImplementedError

    
    def cross_entropy(self, Y, Yhat):
        
        loss = np.sum(-np.log(Yhat+1e-20)*Y)
        return loss
#         pass
#         return Y-Yhat
#         raise NotImplementedError
    

    def cross_entropy_prime(self, Y, Yhat):
        
        loss_prime = -Y/(Yhat + 1e-20)
        return loss_prime
#         return Y-Yhat
#         raise NotImplementedError
    
    def loss(self, Y, Yhat):
        self.Y = Y
        self.Yhat = Yhat
#         self.name = name
        if self.name == "cross_entropy":
            return self.cross_entropy(Y, Yhat)
    
    def loss_prime(self):
#         self.name = name
        if self.name == "cross_entropy":
            return self.cross_entropy_prime(self.Y, self.Yhat)