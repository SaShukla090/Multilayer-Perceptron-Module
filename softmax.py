#sigmoid layer
import numpy as np

class softmax_layer:
    def __init__(self, input_size):
        self.input = None
        self.output = None
        self.input_size = input_size
        
    def forward_propagation(self, input_data):
        self.input = input_data
        x_exp = np.exp(self.input)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
#         print(x_sum)
        self.output = x_exp/(x_sum + 1e-20)

        return self.output
    
    def backward_propagation(self, df_output, learning_rate, optimizer, T):
        Dz_x = np.zeros((self.input_size,self.input_size))
        for i in range(self.input_size):
            for j in range(self.input_size):
                if i == j:
                    Dz_x[i,j] = self.output[0,i]*(1-self.output[0,i])
                else:
                    Dz_x[i,j] = -self.output[0,i]*self.output[0,j]
                    
        DL_x = np.dot(df_output,Dz_x)
        return DL_x
        
        
        
        
        