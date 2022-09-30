from Adam import AdamOptim
import numpy as np
class FC_layer:
    def __init__(self,input_size, output_size, weight_inialization = "xavier"):
        self.input = None
        self.output = None
        self.input_size = input_size
        self.output_size = output_size
        self.weight_inialization = weight_inialization
#         self.weights = None
#         self.bias = None
        
        
        self.weights, self.bias = self.weight_init(self.input_size, self.output_size, self.weight_inialization)
        
    def weight_init(self, input_size, output_size, weight_inialization):
        if weight_inialization == "xavier":
            var = 1/input_size
            weights = np.random.normal(0,var ,(input_size,output_size))
            bias = np.zeros((1,output_size))
            return weights, bias
        
        elif weight_inialization == "random":
            weights = np.random.rand(input_size, output_size) - 0.5
            bias = np.random.rand(1, output_size) - 0.5
            return weights, bias
    
    def forward_propagation(self, input_data):
        self.input = input_data.reshape(1,-1)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, df_output, learning_rate, optimizer , T):
        df_input = np.dot(df_output, self.weights.T)
        df_weights = np.dot(self.input.T, df_output)
        
        self.weights -= learning_rate*df_weights
        self.bias -= learning_rate*df_output
        
        if optimizer == "adam" or "Adam":
            opt = AdamOptim(eta=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            self.weights, self.bias = opt.update(t=T, w = self.weights, b = self.bias, dw = df_weights, db = df_output)
#             print(f"weigths = {self.weights} \nBias = {self.bias} ")
        
#         D = {"df_I":df_input,
#              "df_W":df_weights,
#             }
        
        return df_input