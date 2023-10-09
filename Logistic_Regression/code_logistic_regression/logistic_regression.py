########## >>>>>> Jared Adams 865234

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent




import numpy as np
import math
import sys
sys.path.append("..")

from code_misc.utils import MyUtils



class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1

        

    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''

        self.degree = degree
        X = MyUtils.z_transform(X, degree=self.degree)

        if not SGD:
            batch_size = X.shape[0]

        
#Fitting
        X_bias = np.insert(X, 0, 1, axis=1)
        n, d = X_bias.shape
        self.w = np.zeros((d, 1))

        index_list = self._mini_batch_helper(n, mini_batch_size)
        num_of_batches = len(index_list)

        mini_batch_index = 0 
        while iterations > 0:
            start, end = index_list[mini_batch_index]
            
            X_mini = X_bias[start : end]
            y_mini = y[start : end]

            n_mini, _ = X_mini.shape

            s = y_mini * (X_mini @ self.w)
            self.w = (eta / n_mini) * ((y_mini * self._v_sigmoid(-s)).T @ X_mini).T + (1 - (2 * lam * eta / n_mini)) * self.w

            iterations -= 1
            mini_batch_index = (mini_batch_index + 1) % num_of_batches
            


    def _mini_batch_helper(self, n, mini_batch_size):
        num_of_batches = math.ceil(n / mini_batch_size)

        index_list = []

        for i in range(num_of_batches):
            start_index = i * mini_batch_size
            end_index = (i + 1) * mini_batch_size
            index_list.append((start_index, end_index))
            
        assert len(index_list) == num_of_batches

        return index_list

    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
    
        Z = MyUtils.z_transform(X, degree=self.degree)  # Z-transform to match self.w dimension
        Z_bias = np.insert(Z, 0, 1, axis=1)

        return self._v_sigmoid(Z_bias @ self.w) 
    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''

        y_hat = self.predict(X)
        misclassified = 0
        for (y_pred, x) in zip(y_hat, y):
            if (y_pred > 0.5 and int(x) == -1) or (y_pred <= 0.5 and int(x) == 1):
                misclassified += 1

        return misclassified
    
    
    @staticmethod
    def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
            
        # Hint: use the np.vectorize API

        v_sigmoid = np.vectorize(LogisticRegression._sigmoid)
        return v_sigmoid(s)
    
    
    @staticmethod
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''

        # remove the pass statement and fill in the code.         
        return 1 / (1 + np.exp(-s))
    
