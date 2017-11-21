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
    plt.savefig('x_and_v_plot_%(a)s_%(b)s_%(c)s_%(d)s.png' % {"a": x_0, "b": v_0, "c": h, "d": t})
    plt.close()


def euler_err(h, t):
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
    plt.savefig('x_err_and_v_err_plot_%(a)s_%(b)s_%(c)s_%(d)s.png' % {"a": 1, "b": 0, "c": h, "d": t})
    plt.close()


def maxabs(a, axis=None):
    """Return slice of a, keeping only those values that are furthest away
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
    "Plotting max x error for h0/2^j; j = 0,1,2,...,k when solving up to t=20"""
    err_array = np.zeros(k+1)
    h_array = np.zeros(k+1)

    for i in range(k):
        err_array[i] = max_x_err(float(h0/(2**i)))
        h_array[i] = float(h0/(2**i))

    plt.xlabel('h')
    plt.ylabel('$max[x_{analytic}(t_i) - x_i]$')
    plt.plot(h_array, err_array)
    plt.savefig('err_behavior_%(a)s_%(b)s_%(c)s.png' % {"a": h0, "b": k, "c": 20})
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
    plt.savefig('energy_%(a)s_%(b)s_%(c)s_%(d)s.png' % {"a": x_0, "b": v_0, "c": h, "d": t})
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
        x_arr[i + 1] = float(1/(h**2 + 1))*(float(x_arr[i]) + float(h)*float(v_arr[i]))
        v_arr[i + 1] = float(1/(h**2 + 1))*(float(v_arr[i]) - float(h)*float(x_arr[i]))

    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(t_arr, x_arr)

    plt.subplot(212)
    plt.xlabel('t')
    plt.ylabel('v')
    plt.plot(t_arr, v_arr)
    plt.savefig('x_and_v_plot_implicit_%(a)s_%(b)s_%(c)s_%(d)s.png' % {"a": x_0, "b": v_0, "c": h, "d": t})
    plt.close()


def euler_implicit_err(h, t):
    N = int(t/h)
    x_arr = np.zeros(N + 1)
    v_arr = np.zeros(N + 1)
    t_arr = np.arange(N + 1, dtype=float)

    t_arr *= float(h)

    x_arr[0] = 1
    v_arr[0] = 0

    for i in range(len(t_arr) - 1):
        x_arr[i + 1] = float(1/(h**2 + 1))*(float(x_arr[i]) + float(h)*float(v_arr[i]))
        v_arr[i + 1] = float(1/(h**2 + 1))*(float(v_arr[i]) - float(h)*float(x_arr[i]))

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
    plt.savefig('implicit_x_err_and_v_err_plot_%(a)s_%(b)s_%(c)s_%(d)s.png' % {"a": 1, "b": 0, "c": h, "d": t})
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
        x_arr[i + 1] = float(1/(h**2 + 1))*(float(x_arr[i]) + float(h)*float(v_arr[i]))
        v_arr[i + 1] = float(1/(h**2 + 1))*(float(v_arr[i]) - float(h)*float(x_arr[i]))

    x_err = np.zeros(N + 1)

    for i in range(len(t_arr)):
        x_err[i] = np.cos(t_arr[i]) - x_arr[i]

    return maxabs(x_err)


def implicit_err_behavior(h0, k):
    "Plotting max x error for h0/2^j; j = 0,1,2,...,k when solving up to t=20"""
    err_array = np.zeros(k+1)
    h_array = np.zeros(k+1)

    for i in range(k):
        err_array[i] = implicit_max_x_err(float(h0/(2**i)))
        h_array[i] = float(h0/(2**i))

    plt.xlabel('h')
    plt.ylabel('$max[x_{analytic}(t_i) - x_i]$')
    plt.plot(h_array, err_array)
    plt.savefig('implicit_err_behavior_%(a)s_%(b)s_%(c)s.png' % {"a": h0, "b": k, "c": 20})
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
        x_arr[i + 1] = float(1/(h**2 + 1))*(float(x_arr[i]) + float(h)*float(v_arr[i]))
        v_arr[i + 1] = float(1/(h**2 + 1))*(float(v_arr[i]) - float(h)*float(x_arr[i]))

    energy_arr = x_arr**2 + v_arr**2

    plt.xlabel('t')
    plt.ylabel('E')
    plt.plot(t_arr, energy_arr)
    plt.savefig('implicit_energy_%(a)s_%(b)s_%(c)s_%(d)s.png' % {"a": x_0, "b": v_0, "c": h, "d": t})
    plt.close()


implicit_err_behavior(0.01, 4)

