import scipy.integrate as integrate
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


#  We create an array of the errors for our numerical integrators and plot each from stepnumber M to N
def err_plot_large(M, N):
    n_init_list = np.arange(N + 1)
    n_step_list = np.delete(n_init_list, [0])[M:]
    trap_error_list = np.zeros(len(n_step_list))
    for i in range(len(n_step_list)):
        trap_error_list[i] = -(np.e - 1 - int_trap(exp, 0, 1, n_step_list[i]))

    simp_error_list = np.zeros(len(n_step_list))
    for i in range(len(n_step_list)):
        simp_error_list[i] = -(np.e - 1 - int_simp(exp, 0, 1, n_step_list[i]))

    plt.xlabel('1/N')
    plt.ylabel(r'$\delta x^{1/2}$')
    plt.plot(1.0/n_step_list, np.abs(trap_error_list)**(1.0/2))
    plt.savefig('trap_err_%(a)s_%(b)s.png' % {"a": M, "b": N})
    plt.close()

    plt.xlabel('1/N')
    plt.ylabel(r'$\delta x^{1/4}$')
    plt.plot(1.0/n_step_list, np.abs(simp_error_list)**(1.0/4))
    plt.savefig('simp_err_%(a)s_%(b)s.png' % {"a": M, "b": N})
    plt.close()


def err_plot(N):
    n_init_list = np.arange(N + 1)
    n_step_list = np.delete(n_init_list, [0])
    trap_error_list = np.zeros(N)
    for i in range(len(n_step_list)):
        trap_error_list[i] = -(np.e - 1 - int_trap(exp, 0, 1, n_step_list[i]))

    simp_error_list = np.zeros(len(n_step_list))
    for i in range(len(n_step_list)):
        simp_error_list[i] = -(np.e - 1 - int_simp(exp, 0, 1, n_step_list[i]))

    plt.xlabel('1/N')
    plt.ylabel(r'$\delta x^{1/2}$')
    plt.plot(1.0/n_step_list, np.abs(trap_error_list)**(1.0/2))
    plt.savefig('trap_err_%s.png' % N)
    plt.close()

    plt.xlabel('1/N')
    plt.ylabel(r'$\delta x^{1/4}$')
    plt.plot(1.0/n_step_list, np.abs(simp_error_list)**(1.0/4))
    plt.savefig('simp_err_%s.png' % N)
    plt.close()


def int_simp_specified(func, a, b, acc):
    # Integrate externally defined function using int_simp until specified accuracy acc is reached.
    int_current = 0
    int_next = 0
    k = 1
    N_0 = 10
    while True:
        int_current = int_simp(func, a, b, (N_0)**k)
        int_next = int_simp(func, a, b, (N_0)**(k + 1))
        k += 1
        if np.abs((int_next - int_current)/int_current) < acc:
            return int_current


acc = 0.01

integral1 = int_simp_specified(exp, 0, 1, acc)
expected1 = np.e - 1
err1 = int_simp_specified(exp, 0, 1, acc) - expected1
quad_int1 = integrate.quad(exp, 0, 1)[0]
quad_err1 = integrate.quad(exp, 0, 1)[1]
romberg_int1 = integrate.romberg(exp, 0, 1)
romberg_err1 = romberg_int1 - expected1
print('Integral: e^x from 0 to 1')
print('Expected Value = %s' % expected1)
print('Accuracy = %s' % acc)
print('Value = %s' % integral1)
print('Error = %s' % err1)
print('scipy quad result = %s' % quad_int1)
print('quad error = %s' % quad_err1)
print('scipy romberg result = %s' % romberg_int1)
print('romberg error = %s' % romberg_err1)


def xsq(x):
    return x**2


integral2 = int_simp_specified(xsq, 0, 1, acc)
expected2 = float(1./3.)
err2 = integral2 - expected2
quad_int2 = integrate.quad(xsq, 0, 1)[0]
quad_err2 = integrate.quad(xsq, 0, 1)[1]
romberg_int2 = integrate.romberg(xsq, 0, 1)
romberg_err2 = romberg_int2 - expected2
print('')
print('Integral: x^2 from 0 to 1')
print('Expected value = %s' % expected2)
print('Accuracy = %s' % acc)
print('Value =  %s' % integral2)
print('Error = %s' % err2)
print('scipy quad result = %s' % quad_int2)
print('quad error = %s' % quad_err2)
print('scipy romberg result = %s' % romberg_int2)
print('romberg error = %s' % romberg_err2)

