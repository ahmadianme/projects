import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from utilities import *
import os
import inspect
from pprint import pprint
from mpl_toolkits.mplot3d.art3d import juggle_axes



controlPointCount = 20
surfaceDistance = 2
x = np.zeros((controlPointCount, controlPointCount))
y = np.zeros((controlPointCount, controlPointCount))
z = np.repeat(2., controlPointCount * controlPointCount).reshape((controlPointCount, controlPointCount))
zCurve = np.repeat(0., controlPointCount * controlPointCount).reshape((controlPointCount, controlPointCount))



for i in range(len(x)):
    for j in range(len(x)):
        x[i, j] = i
        y[i, j] = j


x /= controlPointCount
y /= controlPointCount



def bezier(x, y, z, zCurve):
    uPTS = controlPointCount
    wPTS = controlPointCount

    uCELLS = controlPointCount
    wCELLS = controlPointCount

    n = uPTS - 1
    m = wPTS - 1





    u = np.linspace(0, 1, uCELLS)
    w = np.linspace(0, 1, wCELLS)

    b = []
    d = []

    xBezier = np.zeros((uCELLS, wCELLS))
    yBezier = np.zeros((uCELLS, wCELLS))
    zBezier = np.zeros((uCELLS, wCELLS))






    def Ni(n, i):
        return np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))

    def Mj(m, j):
        return np.math.factorial(m) / (np.math.factorial(j) * np.math.factorial(m - j))

    # Bernstein Basis Polynomial
    def J(n, i, u):
        return np.matrix(Ni(n, i) * (u ** i) * (1 - u) ** (n - i))

    def K(m, j, w):
        return np.matrix(Mj(m, j) * (w ** j) * (1 - w) ** (m - j))






    for i in range(uPTS):
        for j in range (wPTS):
            b.append(J(n, i, u))
            d.append(K(m, j, w))

            Jt = J(n, i, u).transpose()

            xBezier = Jt * K(m, j, w) * x[i, j] + xBezier
            yBezier = Jt * K(m, j, w) * y[i, j] + yBezier
            zBezier = Jt * K(m, j, w) * (z[i, j] - surfaceDistance) + zBezier






    # plt.figure()
    #
    # plt.subplot(121)
    # plt.xlabel('Time [0,1]')
    # plt.ylabel('J_20,i')
    #
    # for i, line in enumerate(b):
    #     plt.plot(u, line.transpose())
    #
    # plt.grid()
    #
    # plt.subplot(122)
    # plt.xlabel('Time [0,1]')
    # plt.ylabel('K_20,j')
    #
    # for i, line in enumerate(b):
    #     plt.plot(w, line.transpose())
    #
    # plt.grid()
    #
    # plt.show()


    return x, y, z, zCurve, xBezier, yBezier, zBezier, b, d
























class DraggableVector:
    def __init__(self, vector: matplotlib.axes.Axes.scatter, fig, ax, pointI, pointJ, x, y, z):
        self.vector = vector
        self.fig = fig
        self.ax = ax
        self.press = None


        self.pointI = pointI
        self.pointJ = pointJ

        self.x = x
        self.y = y
        self.z = z

        self.oldX = None
        self.oldY = None
        self.oldZ = None

        def on_press(event: matplotlib.backend_bases.MouseEvent):
            if event.inaxes != self.vector.axes:
                return

            contains, attrd = self.vector.contains(event)

            if not contains:
                return

            if len(selected) > 0:
                selected.pop()

            print('Selected Control Point: ' + str((self.pointI, self.pointJ)))
            selected.append((self.pointI, self.pointJ))




        self.cidpress = self.vector.figure.canvas.mpl_connect("button_press_event", on_press)










_, _, _, _zCurve, xBezier, yBezier, zBezier, b, d = bezier(x, y, z, zCurve)






selected = []

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

ax.plot_surface(xBezier, yBezier, zBezier)



controlPoints = np.empty_like(z).tolist()
for i in range(controlPointCount):
    for j in range(controlPointCount):
        controlPoints[i][j] = ax.scatter(x[i][j], y[i][j], z[i][j], edgecolors='face')
        DraggableVector(vector=controlPoints[i][j], fig=fig, ax=ax, pointI=i, pointJ=j, x=x, y=y ,z=z)





def on_keypress(event):
    step = 0.05

    if event.key == 'm':
        targets = range(math.floor(controlPointCount/4), math.floor(controlPointCount/4*3))

        for i in targets:
            for j in targets:
                z[i][j] += step * 10


    if len(selected) != 0:
        if event.key == 'right':
            x[selected[0][0]][selected[0][1]] += step
        elif event.key == 'left':
            x[selected[0][0]][selected[0][1]] -= step
        if event.key == 'up':
            y[selected[0][0]][selected[0][1]] += step
        elif event.key == 'down':
            y[selected[0][0]][selected[0][1]] -= step
        if event.key == '1':
            z[selected[0][0]][selected[0][1]] += step
        elif event.key == '2':
            z[selected[0][0]][selected[0][1]] -= step
    else:
        if event.key != 'm':
            print('Please left click on a control point then press arrow keys.')



    _, _, _, _zCurve, xBezier, yBezier, zBezier, b, d = bezier(x, y, z, zCurve)

    ax.cla()

    ax.plot_surface(xBezier, yBezier, zBezier)

    controlPoints = np.empty_like(z).tolist()
    for i in range(controlPointCount):
        for j in range(controlPointCount):
            controlPoints[i][j] = ax.scatter(x[i][j], y[i][j], z[i][j], edgecolors='face')
            DraggableVector(vector=controlPoints[i][j], fig=fig, ax=ax, pointI=i, pointJ=j, x=x, y=y ,z=z)

    fig.canvas.draw()
    fig.canvas.flush_events()






fig.canvas.mpl_connect("key_press_event", on_keypress)

plt.show()
