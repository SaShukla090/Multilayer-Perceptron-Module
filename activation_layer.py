from act_func import activation_function
class Activation_Layer:
    def __init__(self, A_funct = "tanh"):
        self.input = None
        self.output = None
        self.A_funct = A_funct
#         self.input_size = input_size
#         self.output_size = output_size
        self.act_class = activation_function()
        
#         if self.A_funct=="tanh":
#             self.act, self.act_prime =  self.act_class

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.act_class.activation(name=self.A_funct, x=input_data)
        return self.output
    
    def backward_propagation(self, df_output, learning_rate, optimizer, T):
        return self.act_class.activation_prime(x=self.input)*df_output