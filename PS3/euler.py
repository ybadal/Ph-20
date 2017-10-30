import numpy as np
import matplotlib.pyplot as plt


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
