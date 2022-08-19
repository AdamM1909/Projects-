# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:21:23 2022

@author: adammyers
"""

class LinearRegression():

    def __init__(self, lr, iterations, SGD_mini_batch_size=None, regularization=None, lam=None):
        self.lr = lr
        self.lam = lam
        self.iterations = iterations
        self.SGD_mini_batch_size = SGD_mini_batch_size
        self.regularization = regularization
        self.lam = lam

    def fit(self, X, y):
        self.m, self.n = X.shape # m is the number of data points, n is the number of features
        self.W = np.zeros(self.n)
        self.b = 0.
        self.loss = []

      
        if self.SGD_mini_batch_size is not None:

            # SGD is used

            for i in tqdm(range(self.iterations)):
                mb_idx = random.sample(range(self.m), self.SGD_mini_batch_size)
                self.X = X[mb_idx,:]
                self.y = y[mb_idx]
                self.update_weights(self.SGD_mini_batch_size, self.regularization) # use only data points in the minibatch
                self.loss.append(self.mse(X, y))

            self.X = X # reset these for plotting later
            self.y = y

        else:
            # all data points are used
            self.X = X
            self.y = y
            
            for i in tqdm(range(self.iterations)):
                self.update_weights(self.m, self.regularization) # use all the data points
                self.loss.append(self.mse(self.X, self.y))
                
        return self

   

    def update_weights(self, n_points, reg):
        pred = self.preds(self.X)
        dW = - (( 2 * ( self.X.T ).dot( self.y - pred ) )/(n_points))
        db = - (2 * np.sum( self.y - pred ) )/ n_points

        if reg == 'Lasso':
            dW += self.lam

        elif reg == 'Ridge':
            dW += 2*self.lam*np.sqrt(self.W.T.dot(self.W))

        else:
            pass
        
        self.W = self.W - dW*self.lr
        self.b = self.b - db*self.lr

        return self

    def preds(self, x):
        return x.dot(self.W) + self.b

    def mse(self, X, Y):
        pred=self.preds(X)
        return np.mean((pred-Y)**2)

    def chi_squared():
        pass

    def mse_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.loss)
        ax.set_xlabel(f'Iteration')
        ax.set_ylabel('MSE')
       

    def plot_2D(self, feature_number=0):

        line=np.linspace(x[:,feature_number].min(), x[:,feature_number].max()).reshape(-1,1)
        ND_line=np.zeros((line.shape[0],self.n))
        ND_line+=line
        fig, ax = plt.subplots()
        ax.grid()
        ax.plot(ND_line[:,feature_number], self.preds(ND_line))
        ax.scatter(x[:,feature_number], y)
        ax.set_xlabel(f'Feture {feature_number}')
        ax.set_ylabel('Y')

       

def data(x,m,c,lam):
    return x.dot(m) + c + lam*np.random.rand(x.shape[0])

 

if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    import random
    import matplotlib.pyplot as plt

    #create some test data

    n_points=150
    m = np.array([1,1])
    x=np.random.rand(n_points,m.shape[0])
    c= 3
    y=data(x,m,c,0.1)
    #fit a linear regression model to it
    model = LinearRegression(lr=0.2, iterations=100, SGD_mini_batch_size=3, regularization='Ridge', lam=0.001)
    model.fit(x,y)
    print(f'Fitted Weight Vector: {model.W}')
    print(f'Target Weight Vector: {m}')
    print(f'Fitted Bias {model.b}')
    print(f'Target Bias {c}')
    print(f'MSE: {model.mse(x,y)}')
    model.mse_plot()
    model.plot_2D(feature_number=1)

