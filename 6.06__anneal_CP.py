# https://www.ensta-bretagne.fr/jaulin/robmooc.html
from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py


def draw_room():
    for j in range(A.shape[1]):
        plot(array([A[0, j], B[0, j]]), array([A[1, j], B[1, j]]), color='blue')


def draw(p, y, col):
    draw_tank(p, 'darkblue', 0.1)
    p = p.flatten()
    y = y.flatten()
    for i in arange(0, 8):
        plot(p[0] + array([0, y[i] * cos(p[2] + i * pi / 4)]), p[1] + array([0, y[i] * sin(p[2] + i * pi / 4)]),
             color=col)


A = array([[0, 7, 7, 9, 9, 7, 7, 4, 2, 0, 5, 6, 6, 5],
           [0, 0, 2, 2, 4, 4, 7, 7, 5, 5, 2, 2, 3, 3]])
B = array([[7, 7, 9, 9, 7, 7, 4, 2, 0, 0, 6, 6, 5, 5],
           [0, 2, 2, 4, 4, 7, 7, 5, 5, 0, 2, 3, 3, 2]])
y = array([[6.4], [3.6], [2.3], [2.1], [1.7], [1.6], [3.0], [3.1]])


def distance_wall(wall_start, wall_end, pose):

    m = pose[:2]
    theta = pose[2]
    y = np.full([8, 1], np.inf)

    for i in range(1, 9):  # n° sensors
        u = array([cos(theta + (i-1)*pi/4), sin(theta + (i-1)*pi/4)])  # sensor direction

        for j in range(A.shape[1]):
            a = wall_start[:, j].reshape(2, 1)
            b = wall_end[:, j].reshape(2, 1)

            cross_wall = det(np.hstack([a-m, u]))*det(np.hstack([b-m, u]))
            if cross_wall < 0:
                alpha = -det(np.hstack([b-a, m-a])) / det(np.hstack([b-a,u]))
                if alpha > 0:
                    y[i-1] = min(alpha, y[i-1])
    return y

# %%
ax = init_figure(-2, 10, -2, 10)

p0 = array([[1], [1], [0]])  # initial guess
draw_room()


j = np.inf
j0 = norm(y - distance_wall(A, B, p0), 2)
draw(p0, y, 'red')
pause(1)

T = 3

while T > 0.01:
    pe = p0+T*randn(3, 1)
    if np.any(distance_wall(A, B, pe) == np.inf):
        je = np.inf
    else:
        je = norm(y-distance_wall(A, B, pe))
    print(T, je)
    draw_room()
    draw(pe, y, 'red')
    pause(0.1)
    if je < j0:
        j0 = je
        p0 = pe
    T *= 0.99
