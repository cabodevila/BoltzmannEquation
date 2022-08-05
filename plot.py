import numpy as np
import matplotlib.pyplot as plt

import os
import re

os.makedirs('figures/function_gluon', exist_ok=True)
os.makedirs('figures/function_quark', exist_ok=True)

def plot_fun(lattice, steps):

    for n in ['gluon', 'quark']:
        fun_arx = sorted(os.listdir('data/function_' + n),
                         key=lambda x: int(re.search('iteration_(.*).txt', x).group(1)))
        for i, name in enumerate(fun_arx):
            if i % 10 == 0:
                plt.clf()
                plt.plot(lattice, lattice * np.loadtxt('data/function_' + n + '/' + name),
                         label='distribution')
                plt.xscale('log')
                plt.title(name)
                plt.legend()
                plt.grid()
                plt.savefig('figures/function_' + n + '/fun_' + name[:-4] + '.png')

def plot_integrals(data, time_step, steps, data_save):

    Ia = data[:,0]
    Ib = data[:,1]
    Ic = data[:,2]
    T = Ia / Ib

    time = np.loadtxt('data/iterations.txt')
    x_axis = time_step * time


    T_thermal = 0.269915 * 0.1   # It depends on the initial conditions and simulation parameters

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, Ia, label=r'$I_a$')
    plt.plot(x_axis, Ib, label=r'$I_b$')
    plt.plot(x_axis, Ic, label=r'$I_c$')
    plt.plot(x_axis, T, label=r'$T_*$')
    plt.hline(x_axis, T_thermal, 'b--', label=r'$T_th$')

    plt.grid()
    plt.legend(loc='lower left')
    plt.savefig('figures/integrals.png')

    return

def plot_stats(dataG, dataQ, time_step, steps, data_save):

    num = dataG[:,0]
    ene = dataG[:,1]
    ent = dataG[:,2]

    Num = dataQ[:,0]
    Ene = dataQ[:,1]
    Ent = dataQ[:,2]

    time = np.loadtxt('data/iterations.txt')
    x_axis = time_step * time

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, num, 'r-', label='Gluon number')
    plt.plot(x_axis, ene, 'r--', label='Gluon energy')
    plt.plot(x_axis, ent, 'r-.', label='Gluon entropy')

    plt.plot(x_axis, Num, 'g-', label='Quark number')
    plt.plot(x_axis, Ene, 'g--', label='Quark energy')
    plt.plot(x_axis, Ent, 'g-.', label='Quark entropy')

    plt.grid()
    plt.legend()
    plt.savefig('figures/stats.png')

    return

def plot_stats_total(dataG, dataQ, time_step, steps, data_save):

    num = dataG[:,0]
    ene = dataG[:,1]
    ent = dataG[:,2]

    Num = dataQ[:,0]
    Ene = dataQ[:,1]
    Ent = dataQ[:,2]

    time = np.loadtxt('data/iterations.txt')
    x_axis = time_step * time

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, num+Num, 'r-', label='Number')
    plt.plot(x_axis, ene+Ene, 'r--', label='Energy')
    plt.plot(x_axis, ent+Ent, 'r-.', label='Entropy')

    plt.grid()
    plt.legend()
    plt.savefig('figures/stats_global.png')

    return


lattice = np.loadtxt('data/lattice.txt')
integrals = np.loadtxt('data/integrals.txt')
stats = np.loadtxt('data/stats_gluons.txt')
Stats = np.loadtxt('data/stats_quarks.txt')


time_step = 1e-4
steps = int(1e7)
data_save = 1000


plot_fun(lattice, steps)
plot_integrals(integrals, time_step, steps, data_save)
plot_stats(stats, Stats, time_step, steps, data_save)
plot_stats_total(stats, Stats, time_step, steps, data_save)
