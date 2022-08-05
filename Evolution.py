"""
Defines a class to evolve the system
"""

import numpy as np
import matplotlib.pyplot as plt
import Kernel

import os

from Lattice import Lattice

class Evolution(Lattice):

    def __init__(self, deltat, length, p_max, p_newspacing=0,
                 init_params=[0.1, 0.1, 0.01],
                 params=[3], save=1000, plot=1000):

        super().__init__(length, p_max, p_newspacing, init_params, params)

        self.deltat = deltat
        self.save   = save   # Save the data each 'save' iterations
        self.plot   = plot   # Plot the data each 'plot' iterations

        self.Nc = params[0]

        self.pf  = self.lattice * self.function
        self.pf_ = self.lattice_ * self.function_

        # Save the lattice in a text file
        os.mkdir('data')
        self.save_lattice()

        # Get values necesary for the inelastic kernel computation
        self.inelastic_parameters()

        return

    def inelastic_parameters(self):

        """
        Compute some constant parameters needed for the inelastic kernel
        """

        x_ = np.logspace(-5, np.log10(0.5), 1000)
        self.x = np.append(x_, np.flip(1-x_))
        self.factors = [1/self.x, (1-self.x)/self.x, self.x, 1-self.x]

        return

    def next_step(self):

        """
        Compute the necesary values to evolve the system to the next step
        """

        ker = Kernel.Kernel(self.lattice, self.lattice_,
                            self.function, self.pf,
                            params=[self.Nc],
                            init_params=[self.alphas, self.Qs, self.f0],
                            inelastic=[self.x, self.factors])

        self.kernel, self.elastic_kernel, self.inelastic_kernel, self.integrals = ker.kernels()

        volume = 0.5 * (self.lattice_[1:]**2 - self.lattice_[:-1]**2)

        self.pf = self.pf + self.deltat * self.kernel / volume
        self.function = self.pf / self.lattice

        return

    def evolve(self, steps):

        """
        Performs the numerical evolution of the system
        """

        if self.plot != False:
            self.fig, self.axs = plt.subplots(1, 3, figsize=(21,7))

        for i in range(steps):

            self.next_step()

            if self.save != False and i % self.save == 0:
                self.save_results(i)
                print('========== Iteration %i ===========' %i)

            if self.plot != False and i % self.plot == 0 and i != 0:
                self.plot_results(i)


        return

    def save_results(self, iter):

        """
        Saves the data of the current step in different text files
        :param iter: current iteration of the evolution
        :param additional: additional parameters to save. Must be False or a
                           [Ia, Ib, T_star, Jp, deriv] list
        :return:
        """

        number = np.array([self.number(i, self.function, 'gluon')
                           for i in range(len(self.function))])

        os.makedirs('data/function_gluon', exist_ok=True)
        os.makedirs('data/number_gluon', exist_ok=True)
        os.makedirs('data/kern_elastic', exist_ok=True)
        os.makedirs('data/kern_inelastic', exist_ok=True)

        np.savetxt('data/function_gluon/iteration_%i.txt' %iter, self.function)
        np.savetxt('data/number_gluon/iteration_%i.txt' %iter, number)
        np.savetxt('data/kern_elastic/iteration_%i.txt' %iter, self.elastic_kernel)
        np.savetxt('data/kern_inelastic/iteration_%i.txt' %iter, self.inelastic_kernel)

        total_number  = sum(number)
        total_energy  = sum([self.energy(i, self.function, 'gluon')
                             for i in range(len(self.function)-1)])
        total_entropy = sum([self.entropy(i, self.function, 'gluon')
                             for i in range(len(self.function)-1)])

        integrals = open('data/integrals.txt', 'a')
        stats = open('data/stats_gluons.txt', 'a')
        time = open('data/iterations.txt', 'a')

        integrals.write('%.16e %.16e\n' %(*self.integrals, ))
        stats.write('%.16e %.16e %.16e\n' %(total_number, total_energy, total_entropy))
        time.write('%i\n' %iter)

        integrals.close()
        stats.close()
        time.close()


        return

    def plot_results(self, iter):

        self.fig.suptitle('Iteration: %i' %iter, fontsize=16)


        def thermal(p, T):
            return 1 / (np.exp(p / T) - 1)

        time = np.loadtxt('data/iterations.txt')
        stats = np.loadtxt('data/stats_gluons.txt')
        integrals = np.loadtxt('data/integrals.txt')
        ene = stats[-1,1]
        T_theo = (30 * ene / np.pi**2) ** (1/4)

        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[2].clear()

        self.axs[0].plot(self.lattice, self.pf, 'b.-', label='Distribution')
        self.axs[0].plot(self.lattice, self.lattice*thermal(self.lattice, T_theo),
                         'k--', label='thermal')

        self.axs[0].set_xscale('log')
        self.axs[0].grid()
        self.axs[0].legend()

        self.axs[1].plot(time, stats[:,0], 'b-', label='Gluon number')
        self.axs[1].plot(time, stats[:,1], 'b--', label='Gluon energy')
        self.axs[1].plot(time, stats[:,2], 'b-.', label='Gluon entropy')

        self.axs[1].grid()
        self.axs[1].legend()

        self.axs[2].plot(time, integrals[:,0], label=r'$I_a$')
        self.axs[2].plot(time, integrals[:,1], label=r'$I_b$')
        self.axs[2].plot(time, integrals[:,0] / integrals[:,1], label=r'$T_*$')
        self.axs[2].hlines(T_theo, time[0], time[-1], 'k', '--', label=r'$T_{th}$')

        self.axs[2].grid()
        self.axs[2].legend()
        #plt.legend(loc='upper center', title='Iteration: %i\nBlue: gluons\nRed:quarks' %iter)
        plt.pause(0.1)
        #plt.show()

        return
