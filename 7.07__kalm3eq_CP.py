from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from scipy.stats import norm as nm
from numpy.random import multivariate_normal as mnorm

x = array([[0,0]]).T
cov_x = 50*eye(2)
ax = init_figure(-100, 100, -100, 100)
#draw_ellipse(x, cov_x, 0.99, ax, [1,0.8,0.8])
#pause(1)

y = array([[8], [7], [0]])


C = array([[2,3],[3,2],[1,-1]])

cov_b = diag([1, 4, 4])
b_bar = zeros(3)
b = mnorm(b_bar, cov_b, 1).reshape(3,1)

x_bar = array([[1.5],[1.5]])
cov_x = diag([10000, 10000])

y_tilde = y - C@x_bar
cov_y = C@cov_x@C.T+cov_b
K = cov_x@C.T@inv(cov_y)
cov_e = cov_x-K@C@cov_x
x_hat = x_bar + K@y_tilde

draw_ellipse(x_hat, cov_e, 0.99, ax, [0.5,0.5,0.5])

