"""
Creates a class which defines a lattice for calculation
"""

import numpy as np

class Lattice():

    def __init__(self, length, p_max, p_newspacing, init_params, params):

        self.length = length
        self.p_max  = p_max
        self.deltap = p_max / length

        self.alphas = init_params[0]
        self.Qs     = init_params[1]
        self.f0     = init_params[2]

        self.Nc = params[0]
        self.Nf = 0
        self.CF = 0

        self.lattice_ = self.construct(p_newspacing)
        self.lattice  = self.construct2()

        # Gluon distribution functions
        self.function_ = np.array([self.f0/self.alphas]*len(self.lattice_)) * \
                            np.heaviside(self.Qs - self.lattice_, 0)
        self.function  = np.array([self.f0/self.alphas]*(len(self.lattice_)-1)) * \
                            np.heaviside(self.Qs - self.lattice, 0)
        # Quark distribution function
        self.Function_ = np.zeros(len(self.lattice_))
        self.Function  = np.zeros(len(self.lattice))

        return

    def construct(self, p_newspacing):

        """
        Creates the lattice according to the __init__ parameters
        """

        lat = np.arange(p_newspacing, self.p_max, self.deltap)

        if p_newspacing != 0:
            lat_aux = np.arange(0, p_newspacing, self.deltap*0.2)
            return np.concatenate((lat, lat_aux))
        else:
            return lat

    def construct2(self):

        return (self.lattice_[1:] + self.lattice_[:-1]) / 2

    def number(self, i, f, particle):

        """
        Computes the number of particles in the volume of the phase space which
        momentum is in the interval (p_lattice[i], p_lattice[i+1])
        :param i: int, index of the lattice on which we want to compute the number
        :param f: function distribution of momentum
        :param particle: string with the name of particle (gluon or quark)
        :return: float
        """

        if particle == 'gluon':
            return (self.lattice_[i+1]**3 - self.lattice_[i]**3) * \
                    f[i] / (6 * np.pi**2)

    def energy(self, i, f, particle):

        """
        Computes the energy density in the volume of the phase space which momentum
        is in the interval (p_lattice[i], p_lattice[i+1])
        :param i: int, index of the lattice on which we want to compute the number
        :param f: function distribution of momentum
        :param particle: string with the name of particle (gluon or quark)
        :return: float
        """

        if particle == 'gluon':
            return (self.lattice_[i+1]**4 - self.lattice_[i]**4) * \
                    f[i] / (8 * np.pi**2)

    def entropy(self, i, f, particle):

        """
        Computes the entropy in the volume of the phase space which momentum is
        in the interval (p_lattice[i-1], p_lattice[i])
        Notice that in f=0, there is an indetermination in the term f*log(f),
        so we take the limit 0*log(0)=0
        :param i: int, index of the lattice on which we want to compute the number
        :param f: function distribution of momentum
        :param particle: string with the name of particle (gluon or quark)
        :return: float
        """

        if f[i] <= 0:
            return (self.lattice_[i + 1] ** 3 - self.lattice_[i] ** 3) * \
                    (1 + f[i]) * np.log(1 + f[i]) / (6 * np.pi ** 2)
        else:
            return (self.lattice_[i + 1] ** 3 - self.lattice_[i] ** 3) * (
                    (1 + f[i]) * np.log(1 + f[i]) - f[i] * np.log(f[i])) / \
                    (6 * np.pi ** 2)

    def save_lattice(self):

        """
        Save the lattice in a .txt file
        :return:
        """

        np.savetxt('data/lattice.txt', self.lattice)
        np.savetxt('data/lattice_.txt', self.lattice_)
