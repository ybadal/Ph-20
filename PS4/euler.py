import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

def euler_plot(x_0, v_0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr[0] = x_0
    v_arr[0] = v_0

    for i in range(len(t_arr) - 1):
        x_arr[i + 1] = float(x_arr[i]) + float(h)*float(v_arr[i])
        v_arr[i + 1] = float(v_arr[i]) - float(h)*float(x_arr[i])

    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(t_arr, x_arr)

    plt.subplot(212)
    plt.xlabel('t')
    plt.ylabel('v')
    plt.plot(t_arr, v_arr)
    plt.savefig('euler_plot.png')
    plt.close()


def euler_err(x_0, v_0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr[0] = 1
    v_arr[0] = 0

    for i in range(len(t_arr) - 1):
        x_arr[i + 1] = float(x_arr[i]) + float(h)*float(v_arr[i])
        v_arr[i + 1] = float(v_arr[i]) - float(h)*float(x_arr[i])

    x_err = np.zeros(N + 1)
    v_err = np.zeros(N + 1)

    for i in range(len(t_arr)):
        x_err[i] = np.cos(t_arr[i]) - x_arr[i]
        v_err[i] = -np.sin(t_arr[i]) - v_arr[i]

    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('t')
    plt.ylabel('$x_{analytic}(t_i) - x_i$')
    plt.plot(t_arr, x_err)

    plt.subplot(212)
    plt.xlabel('t')
    plt.ylabel('$v_{analytic}(t_i) - v_i$')
    plt.plot(t_arr, v_err)
    plt.savefig('euler_err.png')
    plt.close()


def maxabs(a, axis=None):
    """Return slice of a, keeping only those values that \
            are furthest away
    from 0 along axis"""
    maxa = np.amax(a)
    mina = np.amin(a)
    
    if abs(maxa) > abs(mina):
        out = maxa

    else:
        out = mina

    return out


def max_x_err(h):
    """Maximum x error when solving up to t=20"""
    t = 20
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr[0] = 1
    v_arr[0] = 0

    for i in range(len(t_arr) - 1):
        x_arr[i + 1] = float(x_arr[i]) + float(h)*float(v_arr[i])
        v_arr[i + 1] = float(v_arr[i]) - float(h)*float(x_arr[i])

    x_err = np.zeros(N + 1)

    for i in range(len(t_arr)):
        x_err[i] = np.cos(t_arr[i]) - x_arr[i]
   
    return maxabs(x_err)


def err_behavior(h0, k):
    "Plotting max x error for h0/2^j; j = 0,1,2,...,k \
            when solving up to t=20"""
    err_array = np.zeros(k+1)
    h_array = np.zeros(k+1)

    for i in range(k):
        err_array[i] = max_x_err(float(h0/(2**i)))
        h_array[i] = float(h0/(2**i))

    plt.xlabel('h')
    plt.ylabel('$max[x_{analytic}(t_i) - x_i]$')
    plt.plot(h_array, err_array)
    plt.savefig('err_behavior.png')
    plt.close()


def euler_energy(x_0, v_0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr[0] = x_0
    v_arr[0] = v_0

    for i in range(len(t_arr) - 1):
        x_arr[i + 1] = float(x_arr[i]) + float(h)*float(v_arr[i])
        v_arr[i + 1] = float(v_arr[i]) - float(h)*float(x_arr[i])

    energy_arr = x_arr**2 + v_arr**2

    plt.xlabel('t')
    plt.ylabel('E')
    plt.plot(t_arr, energy_arr)
    plt.savefig('euler_energy.png')
    plt.close()


def euler_implicit_plot(x_0, v_0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr[0] = x_0
    v_arr[0] = v_0

    for i in range(len(t_arr) - 1):
        x_arr[i + 1] = float(1/(h**2 + 1))*(float(x_arr[i]) \
                + float(h)*float(v_arr[i]))
        v_arr[i + 1] = float(1/(h**2 + 1))*(float(v_arr[i]) \
                - float(h)*float(x_arr[i]))

    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(t_arr, x_arr)

    plt.subplot(212)
    plt.xlabel('t')
    plt.ylabel('v')
    plt.plot(t_arr, v_arr)
    plt.savefig('euler_implicit_plot.png')
    plt.close()


def euler_implicit_err(x0, v0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr[0] = 1
    v_arr[0] = 0

    for i in range(len(t_arr) - 1):
        x_arr[i + 1] = float(1/(h**2 + 1))*(float(x_arr[i]) \
                + float(h)*float(v_arr[i]))
        v_arr[i + 1] = float(1/(h**2 + 1))*(float(v_arr[i]) \
                - float(h)*float(x_arr[i]))

    x_err = np.zeros(N + 1)
    v_err = np.zeros(N + 1)

    for i in range(len(t_arr)):
        x_err[i] = np.cos(t_arr[i]) - x_arr[i]
        v_err[i] = -np.sin(t_arr[i]) - v_arr[i]

    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('t')
    plt.ylabel('$x_{analytic}(t_i) - x_i$')
    plt.plot(t_arr, x_err)

    plt.subplot(212)
    plt.xlabel('t')
    plt.ylabel('$v_{analytic}(t_i) - v_i$')
    plt.plot(t_arr, v_err)
    plt.savefig('euler_implicit_err.png')
    plt.close()


def implicit_max_x_err(h):
    """Maximum x error when solving up to t=20"""
    t = 20
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr[0] = 1
    v_arr[0] = 0

    for i in range(len(t_arr) - 1):
        x_arr[i + 1] = float(1/(h**2 + 1))*(float(x_arr[i]) \
                + float(h)*float(v_arr[i]))
        v_arr[i + 1] = float(1/(h**2 + 1))*(float(v_arr[i]) \
                - float(h)*float(x_arr[i]))

    x_err = np.zeros(N + 1)

    for i in range(len(t_arr)):
        x_err[i] = np.cos(t_arr[i]) - x_arr[i]

    return maxabs(x_err)


def implicit_err_behavior(h0, k):
    "Plotting max x error for h0/2^j; j = 0,1,2,...,k \
            when solving up to t=20"""
    err_array = np.zeros(k+1)
    h_array = np.zeros(k+1)

    for i in range(k):
        err_array[i] = implicit_max_x_err(float(h0/(2**i)))
        h_array[i] = float(h0/(2**i))

    plt.xlabel('h')
    plt.ylabel('$max[x_{analytic}(t_i) - x_i]$')
    plt.plot(h_array, err_array)
    plt.savefig('implicit_err_behavior.png')
    plt.close()


def implicit_euler_energy(x_0, v_0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr[0] = x_0
    v_arr[0] = v_0

    for i in range(len(t_arr) - 1):
        x_arr[i + 1] = float(1/(h**2 + 1))*(float(x_arr[i]) \
                + float(h)*float(v_arr[i]))
        v_arr[i + 1] = float(1/(h**2 + 1))*(float(v_arr[i]) \
                - float(h)*float(x_arr[i]))

    energy_arr = x_arr**2 + v_arr**2

    plt.xlabel('t')
    plt.ylabel('E')
    plt.plot(t_arr, energy_arr)
    plt.savefig('implicit_euler_energy.png')
    plt.close()


def euler_phase(x_0, v_0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)

    x_arr[0] = x_0
    v_arr[0] = v_0

    for i in range(len(x_arr) - 1):
        x_arr[i + 1] = float(x_arr[i]) + float(h)*float(v_arr[i])
        v_arr[i + 1] = float(v_arr[i]) - float(h)*float(x_arr[i])

    plt.xlabel('x')
    plt.ylabel('v')
    plt.plot(x_arr, v_arr)
    plt.axes().set_aspect('equal')
    plt.savefig('euler_phase.png')
    plt.close()


def implicit_euler_phase(x_0, v_0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)

    x_arr[0] = x_0
    v_arr[0] = v_0

    for i in range(len(x_arr) - 1):
        x_arr[i + 1] = float(1/(h**2 + 1))*(float(x_arr[i]) \
                + float(h)*float(v_arr[i]))
        v_arr[i + 1] = float(1/(h**2 + 1))*(float(v_arr[i]) \
                - float(h)*float(x_arr[i]))

    plt.xlabel('x')
    plt.ylabel('v')
    plt.plot(x_arr, v_arr)
    plt.axes().set_aspect('equal')
    plt.savefig('implicit_euler_phase.png')
    plt.close()


def symplectic_euler_phase(x_0, v_0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)

    x_arr[0] = x_0
    v_arr[0] = v_0

    for i in range(len(x_arr) - 1):
        x_arr[i + 1] = float(x_arr[i]) + float(h)*float(v_arr[i])
        v_arr[i + 1] = float(v_arr[i]) - float(h)*float(x_arr[i+1])

    plt.xlabel('x')
    plt.ylabel('v')
    plt.plot(x_arr, v_arr)
    plt.axes().set_aspect('equal')
    plt.savefig('symplectic_euler_phase.png')
    plt.close()


def analytic_phase(x_0, v_0, h, t):
    N = int(t/h)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr = np.cos(t_arr)
    v_arr = -np.sin(t_arr)

    plt.xlabel('x')
    plt.ylabel('v')
    plt.plot(x_arr, v_arr)
    plt.axes().set_aspect('equal')
    plt.savefig('analytic_phase.png')
    plt.close()


def symplectic_euler_energy(x_0, v_0, h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    x_arr[0] = x_0
    v_arr[0] = v_0
    t_arr *= float(h)

    for i in range(len(x_arr) - 1):
        x_arr[i + 1] = float(x_arr[i]) + float(h)*float(v_arr[i])
        v_arr[i + 1] = float(v_arr[i]) - float(h)*float(x_arr[i+1])

    energy_arr = x_arr**2 + v_arr**2

    plt.xlabel('t')
    plt.ylabel('E')
    plt.plot(t_arr, energy_arr)
    plt.savefig('symplectic_euler_energy.png')
    plt.close()


def main():
    if not (sys.argv[1] == 'err_behavior' or \
            sys.argv[1] == 'implicit_err_behavior'):
        eval('%(a)s(%(b)s,%(c)s,%(d)s,%(e)s)' % {"a": sys.argv[1], \
                "b": sys.argv[2], "c": sys.argv[3], "d": sys.argv[4], \
                "e": sys.argv[5]})
    else:
        eval('%(a)s(%(b)s,%(c)s)' % {"a": sys.argv[1], \
                "b": sys.argv[2], "c": sys.argv[3]})


main()
