from network import *
from compute import *
import pandas as pd


class NetworkManager:
    """
    This class works as an interface to compute networks.
    """
    def __init__(self, species=26, patches=5, randomSeed=0, weightSeed=1,
                 initial='random', growth=1, n0=1, alpha0=0.2, N0=None, Alpha0=None,
                 miu_n0=1, sgm_n0=0.05, miu_alpha=0.5, sgm_alpha=0.01,
                 connectance=0.6, sgm_aij=0.1, sgm_qx=10, rho=0.1, Adiag=1, dispersal=0.1, kappa=1,
                 method_ode=2, method_alpha='Softmax', dt=1e-4, maxIteration=100e4, maxError=1e-4,
                 ini_dpd=False, runtime=0,
                 ):
        """
        The parameters of the network.
        :param species: number of species
        :param patches: number of patches
        :param randomSeed: seed of random numbers
        :param growth: intraspecific interaction strength
        :param n0: initial N
        :param initial: initial mode.
            'random'--randomly spawn A and M; 'fixed'--spawn M depending on A to get fixed point (1,1,...,1)T
        :param var_n0: the variance of N0
        :param dispersal: dispersal rate
        :param dt: time interval
        :param maxIteration: max iteration time
        :param maxError: max error in iteration
        :param connectance:  Proportion of realized interactions among all possible ones
        :param var_aij: the variance of the distribution N(0, var) of a_ij
        :param var_qx: the variance of the distribution N(1, var) of q_ij
        """
        self.config = {NetParam.randomSeed: randomSeed,
                       NetParam.weightSeed: weightSeed,
                       NetParam.species: species,
                       NetParam.patches: patches,

                       NetParam.initial: initial,
                       NetParam.N0: N0,
                       NetParam.Alpha0: Alpha0,
                       NetParam.growth: growth,
                       NetParam.n0: n0,
                       NetParam.alpha0: alpha0,
                       NetParam.miu_n0: miu_n0,
                       NetParam.sgm_n0: sgm_n0,
                       NetParam.miu_alpha: miu_alpha,
                       NetParam.sgm_alpha: sgm_alpha,

                       NetParam.connectance: connectance,
                       NetParam.sgm_aij: sgm_aij,
                       NetParam.sgm_qx: sgm_qx,
                       NetParam.rho: rho,
                       NetParam.Adiag: Adiag,
                       NetParam.dispersal: dispersal,
                       NetParam.kappa: kappa,

                       NetParam.method_alpha: method_alpha,
                       NetParam.method_ode: method_ode,
                       NetParam.dt: dt,
                       NetParam.maxIteration: maxIteration,
                       NetParam.maxError: maxError,
                       }

        self.net: RandomNetwork | None = None

    def computeNet(self):
        """
        Compute the network and plot figures.
        :return:
        """
        self.net = RandomNetwork(self.config)
        cpt = Computer(self.net)
        cpt.compute()
        # self.net.plotAll()

    def fixCchangeW(self):
        maxEigvals = []
        for wseed in range(5):
            self.config[NetParam.weightSeed] = wseed
            self.computeNet()
            maxEigvals.append(self.net.maxEigval)
        print(maxEigvals)


if __name__ == '__main__':
    net_manager = NetworkManager()
    # net_manager.computeNet()
    net_manager.fixCchangeW()
