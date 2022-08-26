# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 12:04:42 2022

@author: adammyers
"""


class PCA():
    def __init__(self, sample, min_variance_in_PCA):

        self.sample = sample
        self.mean = sample.mean(axis=0)
        self.min_variance_in_PCA = min_variance_in_PCA

    def fit(self):
        #standardise sample
        standardized = (self.sample-self.mean).T
        self.evalues_sorted, self.evecs_sorted = self.eigen(standardized)
        self.pca_var = self.evalues_sorted/self.evalues_sorted.sum()

        #choose principle components which describe the required variance           
        k = 0
        sum_of_k_ev = 0
        while (sum_of_k_ev / self.evalues_sorted.sum()) < self.min_variance_in_PCA:
            sum_of_k_ev = sum_of_k_ev + self.evalues_sorted[k]
            k = k + 1
        self.PCs = self.evecs_sorted[:,0:k]
        
        #project our data on selected principal components
        self.projection_data = self.PCs.T.dot(standardized).T
        self.projection_matrix = self.PCs.dot(self.PCs.T)
        self.reconstructed_data = standardized.T.dot(self.projection_matrix)+self.mean
        return self

    def eigen(self, standardized):
        cov = np.cov(standardized)
        evalues, evecs = np.linalg.eig(cov)
        order = np.argsort(evalues)[::-1] # reverse sort the eigenvalues 
        evalues_sorted, evecs_sorted = evalues[order], evecs[:,order]
        return evalues_sorted, evecs_sorted
    
def data(x,m,c,lam):
    return x.dot(m) + c + lam*np.random.rand(x.shape[0])

if __name__ == "__main__":
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    
    n_points=30
    m = np.array([8,9,6])
    x=np.random.rand(n_points,m.shape[0])
    y=data(x,m,m.shape[0],1)
    x.shape,y.reshape(30,1).shape
    fig, ax = plt.subplots()
    ax.scatter(x[:,0],y)
    
    sample=np.append(x,y.reshape(30,1), axis=1)
    pca = PCA(sample,1.)
    pca.fit()
    rd = pca.reconstructed_data
    fig, ax = plt.subplots()
    ax.scatter(x[:,2],y)
    ax.scatter(rd[:,2],y)
    plt.show()
    print('okay')