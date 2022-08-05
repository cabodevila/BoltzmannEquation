"""
Computes the 2 elastic collision kernels for a system of quarks and gluons
"""

import numpy as np

from Elastic import elastic_kernel
from Inelastic import inelastic_kernel

class Kernel():

    def __init__(self, lattice, lattice_, function, pf, params, init_params,
                 inelastic):

        self.lattice  = lattice
        self.lattice_ = lattice_

        self.pf  = pf

        self.Nc = params[0]
        self.alphas = init_params[0]
        self.Qs     = init_params[1]
        self.f0     = init_params[2]

        self.x, self.factors = inelastic

        Ia_integrand = self.pf * (lattice + self.pf)
        self.Ia = np.trapz(Ia_integrand, x=lattice)

        Ib_integrand = 2 * self.pf
        self.Ib = np.trapz(Ib_integrand, x=lattice)

        self.pf_ = np.interp(self.lattice_[1:-1], self.lattice, self.pf)

        # Usefull quantities

        # Debye mass
        self.mD2 = 2 * self.alphas * self.Nc * self.Ib / np.pi
        # Jet quenching parameter
        self.qhat = 8 * np.pi * self.alphas**2 * self.Nc**2 * self.Ia \
                    #* np.log(self.pt2 / self.mD2)

        return

    def compute_elastic(self):

        ek = elastic_kernel(self.lattice, self.lattice_,
                            self.pf, self.pf_,
                            self.Ia, self.Ib,
                            [self.mD2, self.qhat])

        kernel = ek.kernel()

        return kernel

    def compute_inelastic(self):

        ik = inelastic_kernel(self.lattice, self.lattice_,
                              self.pf, self.pf_,
                              self.Ia, self.Ib,
                              [self.alphas, self.Qs, self.f0],
                              [self.Nc],
                              [self.mD2, self.qhat],
                              [self.x, self.factors])

        kernel = ik.kernel()

        return kernel

    def kernels(self):

        elastic   = self.compute_elastic()
        inelastic = self.compute_inelastic()

        total_kernel = elastic + inelastic

        return total_kernel, elastic, inelastic, [self.Ia, self.Ib]
