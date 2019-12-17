import numpy as np
import matplotlib.pyplot as plt

class SingleVariableLinearRegressor:
    def __init__(self,X,Y,alpha=0.1):
        
        self.x=X
        self.y=Y
        self.alpha=alpha
        self.theta = [0, 1]
        self.m =len(self.x)
        
    def h(self, x):
        return self.theta[0] + self.theta[1]*x


    def J(self):
        cost=0
        for i in range(self.m):
            cost += ( self.h( self.x[ i ] ) - self.y[i] )**2 
        cost /= ( 2 * self.m )
        return cost
    
    
    def gradientDescent(self):
        
        flag=10000
        
        iteration=0
        
        while flag>0.0001:
            old_cost=self.J()
            
            
            summation = 0
            for i in range(self.m):
                summation += ( self.h( self.x[i] ) - self.y[i] )
            summation *= (self.alpha/self.m)
            
            temp_theta0 = self.theta[0] - summation
            
            
            summation = 0
            for i in range(self.m):
                summation += ( self.h( self.x[i] ) - self.y[i] ) * self.x[i]
            summation *= (self.alpha/self.m)
            temp_theta1 = self.theta[1] - summation
            
            self.theta = [temp_theta0, temp_theta1]
            
            
            new_cost=self.J()
            
            flag=abs(old_cost - new_cost)
            iteration+=1
        print('It converged in {}'.format(iteration))

        
    def plotData(self):
        plt.figure()
        plt.plot(self.x,self.y,'rx')
        plt.show()
        
    def plotHypothesis(self):
        line_x=np.linspace(0,10,100)
        
        plt.figure()
        plt.plot(line_x, self.h(line_x),'b')
        plt.plot(self.x,self.y,'rx')
        plt.show()
    
    
    def predict(self,x):
        for i in x:
            print('Test Case: {}'.format(i))
            print('The Output is {}'.format(self.h(i)))
            
    

        
x=[1,2,3,4,5]
y=[2,1,5,4,6]
regressor = SingleVariableLinearRegressor(x,y)
regressor.plotData()
print('\n\n\n\n')
print(regressor.J())
regressor.gradientDescent()
regressor.plotHypothesis()
regressor.predict([1,3,5,7,8])