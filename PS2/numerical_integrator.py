import numpy as np
import matplotlib.pyplot as plt


def int_trap(func, a, b, N):
    n_list = np.arange(N + 1)
    h_n = (b-a)/float(N)
    x_list = h_n*n_list + np.full(N + 1, a)

    f_list = func(x_list)

    f_list[0] /= 2

    f_list[N] /= 2

    return h_n*np.sum(f_list)


def int_simp(func, a, b, N):
    # We first build an array consisting of the x's (values + averages of adjacent values)
    n_list = np.arange(N + 1)
    h_n = (b-a)/float(N)

    x_list = h_n*n_list + np.full(N + 1, a)

    val_list1 = np.append(np.zeros(1), x_list)
    val_list2 = np.append(x_list, np.zeros(1))
    sum_list = val_list1 + val_list2
    avg_list = np.delete(sum_list, [0, N + 1])
    avg_list /= 2

    f_avg_list = 4*func(avg_list)

    f_x_list = func(x_list)

    f_x_list[0] /= 2
    f_x_list[N] /= 2
    f_x_list *= 2

    f_list = (h_n/6)*(np.append(f_avg_list, f_x_list))

    return np.sum(f_list)


def exp(x):
    return np.e**x


#  We create an array of the errors for our numerical integrators and plot each for different N
def err_plot(N):
    n_init_list = np.arange(N + 1)
    n_step_list = np.delete(n_init_list, [0])
    trap_error_list = np.zeros(N)
    for i in range(len(n_step_list)):
        trap_error_list[i] = -(np.e - 1 - int_trap(exp, 0, 1, n_step_list[i]))

    simp_error_list = np.zeros(len(n_step_list))
    for i in range(len(n_step_list)):
        simp_error_list[i] = -(np.e - 1 - int_simp(exp, 0, 1, n_step_list[i]))

    plt.plot(1.0/n_step_list, trap_error_list**(1.0/2))
    plt.savefig('trap_err.png')
    plt.close()

    plt.plot(1.0/n_step_list, simp_error_list**(1.0/4))
    plt.savefig('simp_err.png')
    plt.close()
