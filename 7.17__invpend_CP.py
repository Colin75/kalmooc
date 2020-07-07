from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
mc,l,g,mr = 5,1,9.81,1
dt = 0.02


def f(x, u):
    s, theta, ds, dtheta = x[0,0], x[1,0], x[2,0], x[3,0]
    dds = (mr*sin(theta)*(g*cos(theta)- l*dtheta**2) + u[0,0])/(mc+mr*sin(theta)**2)
    ddtheta = (sin(theta)*((mr+mc)*g - mr*l*dtheta**2*cos(theta)) + cos(theta)*u[0,0])/ (l*(mc+mr*sin(theta)**2))
    return array([[ds],[dtheta],[dds],[ddtheta]])


A = array([[0, 0, 1, 0],
           [0, 0, 0, 1],
           [0, mr*g/mc, 0, 0],
           [0, (mc+mr)*g/(l*mc), 0, 0]])

B = array([[0, 0, 1/mc, 1/(1*mc)]]).T
poles = array([-2, -2.1, -2.2, -2.3])
K = place(A, B, poles)
E = array([[1, 0, 0, 0]])
h = -inv(E@inv(A-B@K)@B)
C = array([[1, 0, 0, 0],
           [0, 1, 0, 0]])
L = array([])
cov_a = (sqrt(dt)*(10**-3))**2*eye(4)
x = array([[0, 0.2, 0, 0]]).T

ax = init_figure(-3, 3, -3, 3)

for t in arange(0,5,dt):
    w = 2
    u = -K@x + h*w
    a = mvnrnd1(cov_a)
    x = x + dt * f(x, u)
    clear(ax)
    draw_invpend(ax, x)
pause(1)
    