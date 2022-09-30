from loss_func import loss_function
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
class Neural_Network_Classifier:
    def __init__(self,):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.optimizer = None
        self.loss_name =None
        
#         raise NotImplementedError
#     def AdamOptimizer(self, )
#     def Loss_Function(self, loss_name):
#         self.loss_name = loss_name
        
    def add_layer(self,layer):
        self.layers.append(layer)
        
#         raise NotImplementedError
    
    
    def train(self, X, y,X_test, y_test, optimizer = "adam", epoch = 100, learning_rate = 0.01, loss_name = "cross_entropy"):
        self.loss = loss_function(name = loss_name)
        self.optimizer = optimizer
        
        # One Hot encoding of dataset
        en = OneHotEncoder()
        y_test_en = en.fit_transform(y_test.reshape(-1,1))
        y_test_en = y_test_en.toarray()
#         y_test_en
        
        y_en = en.fit_transform(y.reshape(-1,1))
        y_en = y_en.toarray()
        
        
        v_error = []
        vali_err = []
        accuracy_train = []
        accuracy_val = []
        for i in tqdm(range(epoch)):
            
            err = 0
            samples = len(X)
            for j in range(samples):
                output = X[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss.loss(y_en[j], output)
                
#                 print(error)
#                 v_error.append(err)
#             print(err)
            
                error = self.loss.loss_prime()
#                 if j%600 == 0:
#                     print(error)
                for layers in reversed(self.layers):
                    error = layers.backward_propagation(error, learning_rate, optimizer = self.optimizer , T = i+1)
            
            err /= samples
            v_error.append(err)
            
            ## Accuraacy score calculation
            
            y_train_hat = self.predict(X)
            y_train_hat = en.inverse_transform(y_train_hat)
            accuracy_train.append(accuracy_score(y_train_hat, y))
            
            #Validation Set loss
            test_samples = len(X_test)
            err_test = 0
            for j in range(test_samples):
                output = X_test[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err_test += self.loss.loss(y_test_en[j], output)
#             print(err)
            err_test /= test_samples
            vali_err.append(err_test)
            
            ## Accuraacy score calculation
            y_val_hat = self.predict(X_test)
            y_val_hat = en.inverse_transform(y_val_hat)
            accuracy_val.append(accuracy_score(y_val_hat, y_test))
            
            
        
            
        plt.figure(figsize = (15,5))    
        plt.subplot(1,2,1)
        plt.plot(v_error, label = "Training set")
        plt.plot(vali_err, label = "Validation set")
        plt.title("loss_function for learning rate = " + str(learning_rate))
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(accuracy_train, label = "Training set")
        plt.plot(accuracy_val, label = "Validation set")
        plt.title("Accuracy for learning rate = " + str(learning_rate))
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        

        return v_error, vali_err
                
#         raise NotImplementedError
        
    def predict(self, X):
        samples = len(X)
        result = []
        for i in range(samples):
            output = X[i]
            for layer in self.layers:

                output = layer.forward_propagation(output)
            output = (output == np.max(output))*1
            result.append(output[0])
        return result
#             for i in range
        
#         return output
    
        
#         raise NotImplementedError