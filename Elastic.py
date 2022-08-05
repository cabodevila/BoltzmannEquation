import numpy as np

class elastic_kernel():

    def __init__(self, lattice, lattice_, pf, pf_, Ia, Ib, params):

        self.lattice  = lattice
        self.lattice_ = lattice_
        self.pf  = pf
        self.pf_ = pf_

        self.Ia = Ia
        self.Ib = Ib
        self.T = Ia / Ib

        self.mD2  = params[0]
        self.qhat = params[1]

        deltap = (lattice[1:] - lattice[:-1])
        self.pf_deriv = (self.pf[1:] - self.pf[:-1]) / deltap

    def kernel(self):

        deriv = self.lattice_[1:-1] * self.pf_deriv - self.pf_
        ff = self.pf_ * (self.lattice_[1:-1] + self.pf_) / self.T

        Jp_ = self.qhat * (deriv + ff) / 4
        Jp_ = np.insert(Jp_, 0, 0)
        Jp_ = np.append(Jp_, 0)

        deriv = (Jp_[1:] - Jp_[:-1]) / (2 * np.pi**2)

        return deriv
