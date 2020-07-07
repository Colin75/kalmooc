from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from roblibCP import *

A = array([[2,1],[15,5],[3,12]])
B = array([[15,5],[3,12],[2,1]])    
x_hat = array([[1,2]]).T
cov_x = 100 * eye(2)
cov_b = 1
ax=init_figure(-5,20,-5,20)
for i in range(3):  
    pause(0.5)  
    a,b = A[i],B[i]
    plot([a[0],b[0]],[a[1],b[1]],color="black")    
draw_disk(x_hat,0.5,ax,"blue")


def dist_wall(a, b, x):
    u = (b-a)/norm(b-a)
    d = det(np.hstack([u,x-a])) + randn()
    return d


for i in range(3):
    d = dist_wall(A[i][:, None], B[i][:, None], x_hat)
    print(d)


# Model : prediction
A_pred = eye(2)
u = 0
cov_a = zeros([2,2])

# Model : correction
cov_b = eye(3)

d = array([2, 5, 4])
y = (d-(A[:,1]*B[:,0]-A[:,0]*B[:,1])/norm(B-A, axis=1))[:,None]
C = array([A[:,1]-B[:,1], A[:,0]-B[:,0]]).T/norm(B-A,axis=1)[:,None]

# Init
x0 = array([[1, 2]]).T
cov_x0 = 100 * eye(2)

# Filtrage
x_hat, cov_x = kalman_filter(A_pred,C,u,y,x0,cov_x0,cov_b,cov_a)

draw_disk(x_hat,0.5,ax,"blue")
draw_ellipse(x0, cov_x0, 0.90, ax, [0.8, 0.8, 0.8])
draw_ellipse(x_hat, cov_x, 0.90, ax, [0.5, 0.3, 0.3])