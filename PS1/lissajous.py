#!/usr/bin/python


import math
import sys


# Here's a comment to use as change for the git assignment.


# We first define the sinusoid functions:
def x(ax, fx, t):
    return ax*math.cos(2*math.pi*fx*t)


def y(ay, fy, t, phi):
    return ay*math.sin(2*math.pi*fy*t + phi)


def z(ax, fx, ay, fy, t, phi):
    return x(ax, fx, t) + y(ay, fy, t, phi)


# We then define the listing functions for each sinusoid
def listx(ax, fx, delta_t, nmax):
    xlist = []
    for n in range(0, nmax):
        xlist += [x(ax, fx, n*delta_t)]
    return xlist


def listy(ay, fy, phi, delta_t, nmax):
    ylist = []
    for n in range(0, nmax):
        ylist += [y(ay, fy, n*delta_t, phi)]
    return ylist


def listz(ax, fx, ay, fy, phi, delta_t, nmax):
    zlist = []
    for n in range(0, nmax):
        zlist += [z(ax, fx, ay, fy, n*delta_t, phi)]
    return zlist


# Test line
# print listx(float(sys.argv[3]), float(sys.argv[1]), float(sys.argv[6]), int(sys.argv[7]))


# We now take the input constants from the shell and print the lists for X, Y, and Z to different files.
xfile = open('x_list.txt', 'w')
xfile.write("%s" % listx(float(sys.argv[3]), float(sys.argv[1]), float(sys.argv[6]), int(sys.argv[7])))
xfile.close()


yfile = open('y_list.txt', 'w')
yfile.write("%s" % listy(float(sys.argv[4]), float(sys.argv[2]), float(sys.argv[5]), float(sys.argv[6]), int(
    sys.argv[7])))
yfile.close()


zfile = open('z_list.txt', 'w')
zfile.write("%s" % listz(float(sys.argv[3]), float(sys.argv[1]), float(sys.argv[4]), float(sys.argv[2]), float(
    sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])))
zfile.close()
