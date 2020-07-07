from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
import numpy as np
    
tmax=100
ax=init_figure(0,tmax,-10,10)
delta=1
T=arange(0,tmax,delta)
kmax=size(T)
X=randn(1,kmax)
X=X.flatten()
plot(T, X, 'red')
pause(1)

Y = np.zeros(len(T))
for t in range(len(T)):
    Y[t] = Y[t]+delta*X[t]

np.cov(Y[:5000])