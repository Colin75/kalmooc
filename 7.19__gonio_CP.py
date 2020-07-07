from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from roblibCP import *


def f(x,u):
    x=x.flatten()
    u=u.flatten()
    return (array([[x[3]*cos(x[4])*cos(x[2])],
                   [x[3]*cos(x[4])*sin(x[2])],
                   [x[3]*sin(x[4])/3],
                   [u[0]],
                   [u[1]]]))


def g(x, n_mesures):
    x=x.flatten()
    for i in range(La.shape[1]):
        C = array([[0,0,1]])
        y=array([[x[3]]])
        cov_b = diag([1])*1e5
        a=La[:,i].flatten()
        da = a-(x[0:2]).flatten()
        if norm(da) < 15:
            plot(array([a[0],x[0]]),array([a[1],x[1]]),"red",1)
            delta=arctan2(da[1],da[0])
            for _ in range(n_mesures):
                Ci = array([[-sin(delta),cos(delta), 0]])
                C = vstack((C,Ci))
                yi = [[-sin(delta)*a[0] + cos(delta)*a[1]]]
                y = vstack((y,yi))
                cov_b = block_diag(cov_b,1)
    y += mvnrnd1(cov_b)
    return C, y, cov_b


def h(xa, xb):
    xa = xa.flatten()
    xb = xb.flatten()
    C = array([[0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1]])
    y = array([[xa[3]],
               [xb[3]]])
    cov_b = diag([1, 1])
    dr = xa[:2] - xb[:2]
    # Les 2 robots se détectent
    if norm(dr) < 30:
        plot(array([xa[0], xb[0]]), array([xa[1], xb[1]]), "blue", 1)
        delta = arctan2(dr[1], dr[0])
        Cr = array([[-sin(delta), cos(delta), 0, sin(delta), -cos(delta), 0]])
        C = vstack((C, Cr))
        yr = array([0])
        y = vstack((y, yr))
        cov_b = block_diag(cov_b, 1)

    for i in range(La.shape[1]):
        a = La[:, i].flatten()
        da = a - (xa[0:2]).flatten()
        db = a - (xb[0:2]).flatten()
        # Le robot Ra détecte un amer
        if norm(da) < 15:
            plot(array([a[0], xa[0]]), array([a[1], xa[1]]), "red", 1)
            delta = arctan2(da[1], da[0])
            Ci = array([[-sin(delta), cos(delta), 0, 0, 0, 0]])
            C = vstack((C, Ci))
            yi = [[-sin(delta) * a[0] + cos(delta) * a[1]]]
            y = vstack((y, yi))
            cov_b = block_diag(cov_b, 1)
        # Le robot Rb détecte un amer
        if norm(db) < 15:
#            plot(array([a[0], xb[0]]), array([a[1], xb[1]]), "blue", 1)
            delta = arctan2(db[1], db[0])
            Ci = array([[0, 0, 0, -sin(delta), cos(delta), 0]])
            C = vstack((C, Ci))
            yi = [[-sin(delta) * a[0] + cos(delta) * a[1]]]
            y = vstack((y, yi))
            cov_b = block_diag(cov_b, 1)

    y = y + mvnrnd1(cov_b)

    return C, y, cov_b


def onecar(dt, tmax, La, n_mesures=1):
    ua = array([[0], [0]])
    xa = array([[10, -20, pi / 3, 20, 0.1]]).T
    z_hata = array([[0], [0], [0]])
    cov_za = 10 ** 3 * eye(3)
    cov_axa = dt * diag([0.001, 0.001, 0, 0.001, 0])
    cov_aza = dt * diag([0.001, 0.001, 0.001])

#    ax=init_figure(-50,50,-50,50)
    N = int(tmax/dt)

    x_preds = np.zeros([N, z_hata.shape[0], z_hata.shape[1]])
    cov_preds = np.zeros([N, cov_za.shape[0], cov_za.shape[1]])
    x_ups = np.zeros_like(x_preds)
    cov_ups = np.zeros_like(cov_preds)
    A = np.zeros([N, 3, 3])
    y = []

    for k, t in enumerate(arange(0,tmax,dt)):
        theta_a, delta_a = xa[2][0], xa[4][0]
        Ca, ya, cov_b = g(xa, n_mesures)
        Ak = array([[1, 0, dt * cos(delta_a) * cos(theta_a)],
                   [0, 1, dt * cos(delta_a) * sin(theta_a)],
                   [0, 0, 1]])
        uza = dt * array([[0, 0, ua[0][0]]]).T
        z_hata, cov_za, z_up, cov_up = kalman_filter(Ak, Ca, uza, ya, z_hata, cov_za, cov_b, cov_aza)
        x_preds[k] = z_hata
        cov_preds[k] = cov_za
        x_ups[k] = z_up
        cov_ups[k] = cov_up
        A[k] = Ak
        y.append(ya)

        xa = xa + dt * f(xa, ua) + mvnrnd1(cov_axa)

#        clear(ax)
#        scatter(La[0], La[1])
#        draw_ellipse(z_hata[:2], cov_za[:2, :2], 0.9, ax, 'black')
#        draw_car(xa)
#        if t in range(100):
#            draw_car(xa)
#            draw_ellipse(z_hata[:2], cov_za[:2, :2], 0.9, ax, 'black')
#    pause(0.01)

    return x_preds, cov_preds, x_ups, cov_ups, A, y


def twocars(dt, tmax, La, streaming):

    # # Modele 2 robots
    ua = array([[0.1], [0]])
    ub = array([[0], [0]])

    xa = array([[0, -15, pi/3, 20, 0.1]]).T
    xb = array([[0, 5, pi/3, 20, 0.1]]).T

    z_hat = array([[0]*6]).T
    cov_z = 10 ** 3 * eye(6)
    cov_axa = dt * diag([0.001, 0.001, 0, 0.001, 0])
    cov_axb = dt * diag([0.001, 0.001, 0, 0.001, 0])
    cov_az = dt * diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

    lim = 50
    ax=init_figure(-lim, lim, -lim, lim)
    x_res = [z_hat[0][0]]
    cov_x_res = [cov_z[0, 0]]

    for t in arange(0, tmax, dt):
        theta_a, delta_a = xa[2][0], xa[4][0]
        theta_b, delta_b = xb[2][0], xb[4][0]
        C, y, cov_b = h(xa, xb)

        A = array([[1, 0, dt * cos(delta_a) * cos(theta_a), 0, 0, 0],
                   [0, 1, dt * cos(delta_a) * sin(theta_a), 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, dt * cos(delta_b) * cos(theta_b)],
                   [0, 0, 0, 0, 1, dt * cos(delta_b) * sin(theta_b)],
                   [0, 0, 0, 0, 0, 1]
                   ])
        uz = dt * array([[0, 0, ua[0][0], 0, 0, ub[0][0]]]).T
        z_hat, cov_z = kalman_filter(A, C, uz, y, z_hat, cov_z, cov_b, cov_az)

        xa = xa + dt * f(xa, ua) + mvnrnd1(cov_axa)
        xb = xb + dt * f(xb, ub) + mvnrnd1(cov_axb)

        if streaming:
            clear(ax)
            draw_car(xa)
            draw_car(xb)
            draw_ellipse(xa[:2], cov_z[:2, :2], 0.9, ax, 'black')
            draw_ellipse(xb[:2], cov_z[3:5, 3:5], 0.9, ax, 'black')
        else:
            if t in range(100):
                draw_car(xa)
                draw_car(xb)
                draw_ellipse(xa[:2], cov_z[:2, :2], 0.9, ax, 'black')
                draw_ellipse(xb[:2], cov_z[3:5, 3:5], 0.9, ax, 'black')
        scatter(La[0], La[1])

        x_res.append(z_hat[0])
        cov_x_res.append(cov_z[0, 0])
    pause(0.5)
    return x_res, cov_x_res

La = array([[0,15,30,15],
            [25,30,15,20]])

x_preds, cov_preds, x_ups, cov_ups, A, y = onecar(dt=0.1, tmax=15, La=La, n_mesures=1)
x_backs, cov_backs = kalman_smoother(A, x_preds, cov_preds, x_ups, cov_ups)

# cov_dict = {}
# for i in range(5):
#     x_res, cov_res = onecar(dt=0.1, tmax=15, La=La, n_mesures=i*10)
#     cov_dict[i] = cov_res
# for i in cov_dict.values():
#     plot(i)

#x_res, cov_res = twocars(dt=0.1, tmax=3, La=La, streaming=True)
