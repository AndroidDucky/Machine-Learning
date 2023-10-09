# Jared Adams 865234

### Delete every `pass` statement below and add in your own code. 



# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



import numpy as np
import math
import math_util as mu
import nn_layer
from nn_layer import NeuralLayer


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        newLayer = nn_layer.NeuralLayer(d, act)
        self.layers.append(newLayer)
        self.L += 1
    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        for i in range(1, self.L+1):
            bounds = 1/math.sqrt(self.layers[i-1].d)
            self.layers[i].W = np.random.uniform(-bounds, bounds, (self.layers[i-1].d + 1, self.layers[i].d))

        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.

        x = X
        y = Y
        N = mini_batch_size
        
        step = 0
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set. 
        ## for every iteration:
        #### get a minibatch and use it for:
        for t in range(1, iterations):
            if(SGD == True):
                step += mini_batch_size
                step, x, y = self.mini_Batch(X, Y, step, N)
        ######### forward feeding
            x_out = self._forward_feeding(x)
            self.layers[self.L].Delta = 2 * (x_out - y) * self.layers[self.L].act_de(self.layers[self.L].S)
            
            self.layers[self.L].G = np.einsum('ij, ik -> jk', self.layers[self.L - 1].X, self.layers[self.L].Delta) * (1/N)
        
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
            self._back_propagation(y, N)
        ######### use the gradients to update all the weight matrices.
            for i in range(1, (self.L+1)):
                self.layers[i].W = self.layers[i].W - (eta * self.layers[i].G)

    
    def mini_Batch(self, X, Y, step, N):
        n,d = X.shape
        start_index = step - N
        
        if(n - step < 0):
            end_index = N - (n - start_index)
            x = np.concatenate((X[start_index : n], X[0 : end_index]))
            y = np.concatenate((Y[start_index : n], Y[0 : end_index]))
            step = 0
        else:
            x = X[start_index : step]
            y = Y[start_index : step]
        
        return step, x, y


    def _forward_feeding(self,X):
        self.layers[0].X = np.insert(X, 0, 1, axis=1)
        
        for i in range(1, (self.L+1)):
            cur_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            cur_layer.S = prev_layer.X @ cur_layer.W
            cur_layer.X = cur_layer.act(cur_layer.S)
            cur_layer.X = np.insert(cur_layer.X, 0, 1, axis=1)
            
        return self.layers[self.L].X[:, 1:]

    def _back_propagation(self,Y,N):
        for i in range(self.L-1, 0, -1):
            self.layers[i].Delta = self.layers[i].act_de(self.layers[i].S) *  (self.layers[i+1].Delta @ (self.layers[i+1].W).T)[:,1:]
            self.layers[i].G = np.einsum('ij,ik -> jk', self.layers[i-1].X, self.layers[i].Delta) *  (1/N)


    
    
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''
        predictions = self._forward_feeding(X)
        predicted = np.argmax(predictions, axis=1)
        
        return predicted
    
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        
        sample = np.argmax(Y, axis = 1)
        predicted = self.predict(X)
        
        error = np.sum(sample != predicted)
        
        return error/len(sample)
 
