import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from enum import Enum


class NetworkParameter(Enum):
    """
    Define the name of the network parameters.
    """
    species = 'species'  # number of species
    patches = 'patches'  # number of patches
    randomSeed = 'randomSeed'  # seed of random numbers
    dispersal = 'dispersal'  # dispersal rate
    dt = 'dt'  # time interval
    maxIteration = 'maxIteration'  # max iteration time
    maxError = 'maxError'  # max error in iteration
    connectance = 'connectance'  # Proportion of realized interactions among all possible ones
    var_alpha = 'var_alpha'  # the variance of the distribution N(0, var) of alpha_ij
    var_qx = 'var_qx'  # the variance of the distribution N(1, var) of q_ij


class DispersalNetwork:
    def __init__(self, config: dict):
        self.seed = config['randomSeed']
        np.random.seed(self.seed)

        self.S = config['species']  # numbers of species
        self.n = config['patches']  # numbers of patches
        self.c = config['connectance']  # connectance
        self.var_alpha = config['var_alpha']
        self.var_qx = config['var_qx']
        self.d = config['dispersal']  # dispersal rate
        self.dt = config['dt']  # time interval
        self.maxIter = config['maxIteration']  # max iteration time
        self.maxError = config['maxError']  # max error in iteration
        self.change = [config['change'], config[config['change']]]  # the varying parameter [name, value]

        self.N0 = None  # initial N
        self.M, self.A, self.D = None, None, None
        self.Nf = None  # fixed point
        self.J = None  # Jacobian matrix
        self.eigval = None  # eigenvalues of Jacobian matrix
        self.stable = False  # stability
        self.unstableReason = None  # reason of unstable

        self.N_lst, self.xlst = [], []

    def spawn(self):
        """
        spawn A, M, N0\n
        A: normal distribution N(0, 1)\n
        M: (1, 1, ..., 1)
        N0: (1, 1, ..., 1)
        :return:
        """

        # use 1-D matrix
        self.M = np.ones(self.S * self.n)
        self.N0 = 10 * np.ones(self.S * self.n)
        # print(self.N0)

        self._spawn_A(mode=1)
        self._spawn_D()

    def _spawn_A(self, mode=1):
        """
        spawn the matrix A
        :return:
        """

        if mode == 1:
            A_std = -1 * np.identity(self.S) / 10
            for i in range(self.S):
                for j in range(self.S):
                    if i != j:
                        if np.random.random() < self.c:
                            A_std[i, j] = np.random.normal(0, self.var_alpha) / 50

            lst_A = []
            for i in range(self.n):
                # same = np.random.normal(1, self.var_qx, size=(self.S, self.S))
                # lst_A.append(np.multiply(same, A_std))
                lst_A.append(A_std)
            # print(lst_A[1] - lst_A[0])

            self.A = sp.linalg.block_diag(*lst_A)

        elif mode == 2:
            A_std = -1 * np.identity(self.S)
            for i in range(self.S):
                for j in range(self.S):
                    if i < j:
                        if np.random.random() < self.c:
                            A_std[i, j] = np.random.normal(0, self.var_alpha) / 5
                    elif i > j:
                        A_std[i, j] = -A_std[j, i]

            lst_A = []
            for i in range(self.n):
                # same = np.random.normal(1, 0, size=(self.S, self.S))
                # lst_A.append(np.multiply(same, A_std))
                lst_A.append(A_std)

            self.A = sp.linalg.block_diag(*lst_A)


    def _spawn_D(self):
        """
        spawn the matrix D
        :return:d
        """
        temp = np.concatenate([self.d * np.identity(self.S)] * self.n, axis=1)
        self.D = np.concatenate([temp] * self.n, axis=0)

    def ode(self, t=0, N=0):
        """
        the non-dispersal LV equation represented with matrix
        :param N: current number matrix of species
        :return: dN / dt
        """

        # temp1 = np.multiply(N, self.M - self.d + np.matmul(self.A, N))
        # temp2 = self.d / (self.n-1) * np.matmul(self.C, N)
        # return temp1 + temp2

        temp1 = np.multiply(N, self.M + np.matmul(self.A, N))
        temp2 = np.matmul(self.D, N)
        P = np.ones(self.n*self.S)
        temp3 = np.multiply(N, np.matmul(self.D, P))

        return temp1 + temp2 + temp3

    def RK4(self, N, dt):
        """
        4th-order Runge-Kutta method
        :param N: current number matrix of species
        :param dt: time interval
        :return: N_n+1
        """
        k1 = self.ode(N=N)
        k2 = self.ode(N=N + k1 * dt / 2)
        k3 = self.ode(N=N + k2 * dt / 2)
        k4 = self.ode(N=N + k3 * dt)

        return N + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def findFixed(self):
        """
        find the fixed point
        :return:
        """
        N_old = self.N0
        N_new = N_old
        # if all elements in |N_new - N_old| are < maxError, regard the point as a fixed point
        iteration = 0
        while True:  # sum(abs(N_new - N_old) > self.maxError) > 0:
            if iteration % 2 == 0:
                self.xlst.append(iteration)
                self.N_lst.append(N_new.reshape(1, -1))

            if iteration % 10 == 0:
                print(iteration)

            temp = N_new
            N_new = self.RK4(N_old, self.dt)
            N_old = temp
            iteration += 1

            # if some Ni <= 0, the fixed point does not exist
            if sum(N_new <= 0) > 0:
                # N_new = None
                self.unstableReason = 'some specie(s) extinguish'
                break
            # if iteration time exceeds the maximum, the fixed point does not exist
            if iteration > self.maxIter:
                # N_new = None
                self.unstableReason = 'iteration overflow'
                break

        self.Nf = N_new

    def calc_jacobian(self):
        """
        calculate Jacobian matrix
        :return:
        """

        N_diag = np.diag(self.Nf.reshape(self.n*self.S))
        # N_diag = np.diag(self.Nf)
        temp1 = np.matmul(N_diag, self.A)
        P = np.ones(self.n * self.S)
        temp2 = np.diag(self.M + np.matmul(self.A, self.Nf) - np.matmul(self.D, P))
        temp3 = np.matmul(self.D, N_diag)
        self.J = temp1 + temp2 + temp3

    def calc_eigenvalue(self):
        """
        calculate the eigenvalues of Jacobian matrix\n
        if the real part of each eigenvalue is negative, the point is stable
        :return:
        """
        self.eigval = np.linalg.eigvals(self.J)
        if sum(self.eigval.real < 0) == self.S:
            self.stable = True
        else:
            self.stable = False
            self.unstableReason = 'positive real parts of eigenvalues'

    def compute(self):
        """
        compute the fixed point
        :return:
        """
        self.spawn()
        self.findFixed()
        self.calc_jacobian()
        self.calc_eigenvalue()

        # solver = sp.integrate.RK45(self.ode, t0=0, y0=self.N0, t_bound=10, max_step=self.maxIter)
        # for i in range(int(self.maxIter)):
        #     solver.step()
        #     if solver.status == 'finished':
        #         break
        # self.Nf = solver.y.T

    def disp(self):
        """
        display network information
        :return:
        """
        print(f'Num. Species:{self.S}')
        print(f'Seed: {self.seed}')
        print(f'Fixed point: {self.Nf}')
        print(f'Stable: {self.stable}')
        print(f'Reason: {self.unstableReason}')
        # print(f'A: {self.A}')

    def ploteigval(self, ax: plt.Axes):
        """
        plot the eigenvalues of Jacobian matrix on the complex plane
        :return:
        """
        x = self.eigval.real
        y = self.eigval.imag

        ax.scatter(x, y)
        ax.set_title('Eigenvalues of Jacobian matrix')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        # plt.show()

    def plotN(self, ax: plt.Axes):
        ax.plot(self.xlst, np.concatenate(self.N_lst, axis=0))
        ax.set_title(f'Number of species vs. iteration times')
        ax.set_xlabel('iteration')
        ax.set_ylabel('N')
        # plt.show()

    def plotboth(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        self.plotN(axs[0])
        self.ploteigval(axs[1])
        fig.suptitle(f'{self.change[0].value}: {self.change[1]}')
        plt.show()


class NetworkManager:
    """
    This class works as an interface to compute networks.
    """
    def __init__(self, species, patches, randomSeed=1, dispersal=0, dt=1e-2, maxIteration=3e3, maxError=1e-4,
                 connectance=0.2, var_alpha=0.25, var_qx=0.1, change=NetworkParameter.randomSeed):
        """
        The parameters of the network.
        :param species: number of species
        :param patches: number of patches
        :param randomSeed: seed of random numbers
        :param dispersal: dispersal rate
        :param dt: time interval
        :param maxIteration: max iteration time
        :param maxError: max error in iteration
        :param connectance:  Proportion of realized interactions among all possible ones
        :param var_alpha: the variance of the distribution N(0, var) of alpha_ij
        :param var_qx: the variance of the distribution N(1, var) of q_ij
        """
        self.config = {'randomSeed': randomSeed,
                       'species': species,
                       'patches': patches,
                       'dispersal': dispersal,
                       'dt': dt,
                       'maxIteration': maxIteration,
                       'maxError': maxError,
                       'connectance': connectance,
                       'var_alpha': var_alpha,
                       'var_qx': var_qx,
                       'change': change}

    def changeParam(self, param, start, end, step):
        """
        Research the effect of different network parameters.\n
        The parameter vary in range(start, end, step)
        :param param: network parameter
        :param start:
        :param end:
        :param step:
        :return:
        """
        for var in np.arange(start, end, step):
            self.config[param] = var
            self.config['change'] = param
            self.computeNet()

    def computeNet(self):
        """
        Compute the network and plot figures.
        :return:
        """
        net = DispersalNetwork(self.config)
        net.compute()
        # if not net.stable:
        #     continue
        net.disp()
        net.plotboth()


if __name__ == '__main__':
    species = 30
    patches = 20
    net_manager = NetworkManager(species=species, patches=patches)
    net_manager.changeParam(NetworkParameter.randomSeed, 1, 101, 1)
