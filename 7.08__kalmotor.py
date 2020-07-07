from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

C = array([[4,0],[10,1],[10,5],[13,5],[15,3]])
y = array([5,10,8,14,17]).reshape(-1,1)

# hypothesis
x_bar = array([[1],[-1]])
cov_x = diag([4,4])
cov_b = 9*eye(len(y))

y_tilde = y - C@x_bar
cov_y = C@cov_x@C.T+cov_b
K = cov_x@C.T@inv(cov_y)
cov_e = cov_x-K@C@cov_x
x_hat = x_bar + K@y_tilde
y_hat = C@x_hat
r = y-y_hat
