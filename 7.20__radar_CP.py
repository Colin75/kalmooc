from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from roblibCP import  *


def g(x):
    g1 = norm(x[::2,:]-a)**2
    g2 = norm(x[::2,:]-b)**2
    return(array([[g1],[g2]]))


dt = 0.02
cov_a = diag([0,dt,0,dt])
x = array([[2],[1],[2],[0]])

x_hat = array([[0],[1],[15],[0]])
cov_x = 10**4*eye(len(x))
Ak = array([[1,dt,0,0],[0,1-dt,0,0],[0,0,1,dt],[0,0,0,1-dt]])

cov_b = 5*eye(2)

a = array([[0,0]]).T
b = array([[1,0]]).T

ax=init_figure(-5,5,-5,5)
for t in arange(0,5,dt):
    clear(ax)
    Ck = 2*array([[x_hat[0,0]-a[0,0], 0, x_hat[1,0]-a[1,0], 0],
                  [x_hat[0,0]-b[0,0], 0, x_hat[1,0]-b[1,0], 0]])
    y = g(x) + mvnrnd1(cov_b)
    z = y - g(x_hat)+Ck@x_hat
    uk=array([[0]]*4)
    plot(a[0],a[1],'o')
    plot(b[0],b[1],'o')
    plot(x[0,0],x[2,0],'o')
    plot(x_hat[0, 0], x_hat[2, 0], 'o', 'm')
    draw_disk(a,sqrt(y[0,0]),ax,[0.8,0.8,0.8])
    draw_disk(b,sqrt(y[1,0]),ax,[0.8,0.8,0.8])
    draw_ellipse(x_hat[::2,:], cov_x[:2,:2], 0.9, ax, 'red')
    x_hat, cov_x = kalman_filter(Ak, Ck, uk, z, x, cov_x, cov_b, cov_a)
    x = Ak @ x + mvnrnd1(cov_a)
pause(1)