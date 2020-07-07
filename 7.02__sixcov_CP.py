from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

theta = arange(0, 2 * pi, 0.1)
x = cos(theta)
y = sin(theta)
cov1 = eye(2)
X = cov1@array([x, y])
plot2D(X, 'black', 1)

A = array([[4, 1],[1, 3]])
B = sqrtm(A)
print("A=", A)
print("B*B=", B@B)

cov2 = cov1*3
X2 = sqrtm(cov2)@array([x, y])
plot2D(X2, 'black', 1)

A1 = array([[1, 0],[0, 3]])
cov3 = A1@cov2@A1.T+cov1
X3 = sqrtm(cov3)@array([x, y])
plot2D(X3, 'black', 1)

A2 = array([[cos(pi/4), -sin(pi/4)], [sin(pi/4),cos(pi/4)]])
cov4 = A2@cov3@A2.T
X4 = sqrtm(cov4)@array([x, y])
plot2D(X4, 'black', 1)

cov5 = cov4 + cov3
X5 = sqrtm(cov5)@array([x, y])
plot2D(X5, 'black', 1)

cov6 = A2@cov5@A2.T
X6 = sqrtm(cov6)@array([x, y])
plot2D(X6, 'black', 1)

pause(1)
