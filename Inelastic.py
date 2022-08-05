import numpy as np
import scipy.interpolate as scpi

import multiprocessing as mp

class inelastic_kernel():

    def __init__(self, lattice, lattice_, pf, pf_, Ia, Ib, init_params, params,
                 relevant, inelastic):

        self.lattice  = lattice
        self.lattice_ = lattice_
        self.pf  = pf
        self.pf_ = pf_

        self.Ia = Ia
        self.Ib = Ib
        self.T = Ia / Ib

        self.mD2  = relevant[0]
        self.qhat = relevant[1]

        self.alphas = init_params[0]
        self.Qs     = init_params[1]
        self.f0     = init_params[2]
        self.Nc     = params[0]

        deltap = (lattice[1:] - lattice[:-1])
        self.pf_deriv = (self.pf[1:] - self.pf[:-1]) / deltap

        self.lattice_aux = np.append(lattice, 2 * lattice[-1] - lattice[-2])
        self.function_aux = np.append(self.pf, 0)
        self.extr = scpi.InterpolatedUnivariateSpline(self.lattice_aux,
                                                      self.function_aux, ext=3)
        self.pf = self.extr(lattice_)

        # Compute a x grid distributed as the splitting function
        self.x = inelastic[0]
        self.factors = inelastic[1]
        self.spltr = self.split_rate()

        return

    def split_rate(self):

        """
        Computes the rate of a hard gluon with momentum p \sim Q_s to split almost
        collinearly into two gluons with momenta px and p(1-x)
        """

        h0 = (self.alphas * self.Nc / np.pi) * np.sqrt(self.qhat)
        h = (1-self.x+self.x**2)**(5/2) / (self.x-self.x**2)**(3/2)

        return h0 * h # The 1/sqrt(p) is added later

    def new_grid(self, p):

        self.new_f = []

        for i, factor in enumerate(self.factors):
            new_grid = factor * p
            self.new_f.append(self.extr(new_grid) / factor)

        return

    def integrand(self, i, p):

        self.new_grid(p)

        Ca = (self.new_f[0] * (p + self.pf[i]) * (p + self.new_f[1]) -
             self.pf[i] * self.new_f[1] * (p + self.new_f[0])) / self.x**(5/2)

        Cb = (self.pf[i] * (p + self.new_f[2]) * (p + self.new_f[3]) -
             self.new_f[2] * self.new_f[3] * (p + self.pf[i]))

        integrand = np.trapz(self.spltr * (Ca - Cb * np.heaviside(self.x-0.5, 0)), self.x)

        return integrand

    def kernel(self):

        lattice_aux = np.copy(self.lattice_)
        lattice_aux[0] = self.lattice[0]
        # Execute the integration in x in parallel
        # Using Pool
        lista = [[i, p] for i, p in enumerate(lattice_aux)]
        split_lista = np.array_split(lista, 6)
        with mp.Pool(6) as pool:
            integrand = pool.starmap(self.integrand, lista)

        integrand = integrand * lattice_aux**(-3/2)

        # Integrate in momentum in order to compute the derivative
        derivative = np.array(
            [np.trapz(integrand[i:i+2], x=self.lattice_[i:i+2])
            for i in range(1, len(self.lattice))]
            )

        derivative = np.insert(derivative,
                               0,
                               integrand[1] * self.lattice_[1])

        derivative = derivative / (2 * np.pi**2)

        return derivative
