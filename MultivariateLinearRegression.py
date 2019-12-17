# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:41:17 2019

@author: Farzad
"""

import matplotlib.pyplot as plt
import numpy as np

class MultivariateLinearRegressor:
    def __init__( self, X, y, alpha = 0.001 ):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.Theta = np.random.rand( self.X.shape[0], 1 )
        self.Averages = np.average(self.X, axis =1)
        self.Ranges = np.max(self.X, axis=1) - np.min(self.X, axis=1)
        self.X = self.scale()
        
    def scale(self):
        x = np.zeros_like(self.X)
        for i in range(1, x.shape[0]):
            for j in range(x.shape[1]):
                x[i,j] = (self.X[i,j] - self.Averages[i,0]) / self.Ranges[i,0]
        for i in range(x.shape[1]):
            x[0, i ] = 1
            
        return x
    
    
    def h( self, x ):
        return self.Theta.T * x
    
    def J( self ):
        error = self.h( self.X ) - self.y
        return float( ( error * error.T ) / ( 2 * self.X.shape[1]) )
    
    def gradientDescent(self):
        
        flag = 1000
        
        iterations = 0
        iterations_list = [0]
        cost_list = [self.J()]
        
        while flag > 0.001:
            iterations += 1
            old_cost = self.J()
            error = self.h( self.X ) - self.y
            self.Theta = self.Theta - self.X * error.T * (self.alpha / X.shape[1])
            new_cost = self.J()
            iterations_list.append(iterations)
            cost_list.append(new_cost)
            flag = abs( old_cost - new_cost )
            
        plt.figure()
        plt.title('Debugging')
        plt.xlabel('Iterations')
        plt.ylabel('J($\Theta&)')
        plt.plot(iterations_list, cost_list)
        plt.show()
        
        print('Trained in {} iterations'.format(iterations))
            
    def predict( self, x ):
        print( 'The output is: ' )
        print( self.h( x ) )
    
X = np.matrix( '1,1,1,1;50,70,100,90;1,2,2,1', dtype = float )
y = np.matrix( '70,100,120,110', dtype = float )
regressor = MultivariateLinearRegressor( X, y, 0.00001 )
#print(regressor.h(10))
regressor.gradientDescent()
regressor.predict(X[:,1])
