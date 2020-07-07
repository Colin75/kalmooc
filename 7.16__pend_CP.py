from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from roblibCP import *

def draw_pend(theta,col='black'): #inverted pendulum
    plot([0,sin(theta)], [0,-cos(theta)], col, linewidth = 2)
    
def f(x,u):
    theta, dtheta=x[0,0],x[1,0]
    return array([[dtheta],[-sin(theta)+u]])


ax=init_figure(-2,2,-2,2)
dt = 0.1
sigm_x = 0.05
x = array([[0,0]]).T
cov_a = dt * sigm_x**2 * eye(2)
u=0
xr = array([[0,0]]).T
P = eye(2)
sigm_y = 0.1
cov_b = sigm_y**2
C = array([[0, 1]])
for t in arange(0, 10, dt):
    clear(ax)
    y = C@x+sigm_y*randn()
    w = sin(t)
    dw = cos(t)
    ddw = -sin(t)
    draw_pend(w, "red")
    draw_pend(x[0, 0], "black")
    draw_pend(xr[0,0], "green")
    cov_xr = 5**2*P[0,0]
    draw_arc(array([[0],[0]]), array([[sin(xr[0,0])-cov_xr], [-cos(xr[0,0]-cov_xr)]]), 2*cov_xr, 'magenta')
    u = sin(xr[0, 0]) + w - xr[0, 0] + 2 * (dw - xr[1, 0]) + ddw
    A = array([[1, dt],
               [-dt*cos(xr[0,0]),1]])
    v = dt*array([[0],[-sin(xr[0,0])+xr[0,0]*cos(xr[0,0])+u]])
    xr, P = kalman_filter(A, C, v, y, xr, P, cov_b, cov_a)

    x=x+dt*f(x,u)+mvnrnd1(cov_a)
    pause(0.001)

