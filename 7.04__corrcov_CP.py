from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from numpy import cov

# Ellipsoïde de confiance : correcteur

N=1000

x=randn(2,N)
xbar = array([[1],[2]])
cov_x = array([[3,1],[1,3]])
x = xbar + sqrtm(cov_x)@x
ax=init_figure(-10,10,-10,10)
ax.scatter(x[0],x[1], s=2)
draw_ellipse(xbar,cov_x,0.9,ax,[1,0.8,0.8])

x2 = arange(-10,10)
K = cov_x[0][1]/cov_x[1][1]
xhat1 = xbar[0] + K*(x2-xbar[1])
plot(xhat1, x2, c='green')

x1 = arange(-10,10)
K = cov_x[1][0]/cov_x[0][0]
xhat2 = xbar[1] + K*(x1-xbar[0])
plot(x1,xhat2, c='cyan')

pause(1)