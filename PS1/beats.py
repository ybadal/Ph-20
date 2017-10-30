#!/usr/bin/python


import sys
import numpy as np
import matplotlib.pyplot as plt

# We set up a numpy array with the values we want our functions to pass.
arr_x_val = np.arange(int(sys.argv[7]) + 1)*(2*np.pi*float(sys.argv[1]))*float(sys.argv[6])
arr_y_val = (np.arange(int(sys.argv[7]) + 1)*(2*np.pi*float(sys.argv[2])))*float(sys.argv[6]) + float(sys.argv[5])


# We find the values of our functions over each array
arr_x = float(sys.argv[3])*np.cos(arr_x_val)
arr_y = float(sys.argv[4])*np.sin(arr_y_val)
arr_z = arr_x + arr_y


# We now take the input constants from the shell and print the lists for X, Y, and Z to different files.
xfile = open('x_list_np.txt', 'w')
xfile.write("%s" % arr_x)
xfile.close()


yfile = open('y_list_np.txt', 'w')
yfile.write("%s" % arr_y)
yfile.close()


zfile = open('z_list_np.txt', 'w')
zfile.write("%s" % arr_z)
zfile.close()


# We plot Z
plt.xkcd()
plt.xlabel('t')
plt.ylabel('z')
plt.plot(arr_z)
plt.savefig('beats_%(a)s_%(b)s_%(c)s_%(d)s_%(e)s_%(f)s_%(g)s.png' % {"a": sys.argv[1], "b": sys.argv[2],
                                                                         "c": sys.argv[3], "d": sys.argv[4],
                                                                         "e": sys.argv[5], "f": sys.argv[6],
                                                                         "g": sys.argv[7]}, bbox_inches='tight')
