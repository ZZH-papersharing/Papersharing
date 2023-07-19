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
    m = 'growth rate'  # intra-specific interaction strength
    n0 = 'initial N'  # initial N
    initial = 'initial mode'  # initial mode
    var_n0 = 'var_n0'  # the variance of N0
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
        self.m = config['m']  # intraspecific interaction strength
        self.n0 = config['n0']  # initial N
        self.initial = config['initial']  # initial mode
        self.var_n0 = config['var_n0']
        self.var_alpha = config['var_alpha']
        self.var_qx = config['var_qx']
        self.d = config['dispersal']  # dispersal rate
        self.dt = config['dt']  # time interval
        self.maxIter = config['maxIteration']  # max iteration time
        self.maxError = config['maxError']  # max error in iteration
        self.change = [config['change'], config[config['change'].value]]  # the varying parameter [name, value]

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
        self._spawn_A(mode=1)
        self._spawn_D()

        # use 1-D matrix
        if self.initial == 'fixed':
            # let (1,1,...,1)T as the fixed point in non-dispersal situation
            self.M = np.matmul(self.A, -1*np.ones(self.S * self.n))
            self.N0 = np.random.normal(1, self.var_n0, size=self.S * self.n)
        elif self.initial == 'random':
            self.M = self.m * np.ones(self.S * self.n)
            self.N0 = self.n0 * np.ones(self.S * self.n)

    def _spawn_A(self, mode=1):
        """
        spawn the matrix A
        :param mode: 1--random web; 2--food web (might have some problem)
        :return:
        """

        if mode == 1:
            A0 = np.random.normal(0, self.var_alpha, size=(self.S, self.S))
            Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            A_std = np.multiply(A0, Connect)
            A_std -= np.diag(np.diag(A_std) + 1)
            # A_std = -1 * np.identity(self.S)
            # for i in range(self.S):
            #     for j in range(self.S):
            #         if i != j:
            #             if np.random.random() < self.c:
            #                 A_std[i, j] = np.random.normal(0, self.var_alpha)
            # print(A_std0 - A_std)

            # print(A_std == 0)
            # print(sum(A_std==0) / self.S)
            # print(A_std)
            #
            # fig, ax = plt.subplots()
            # im, cbar = heat
            # im = ax.imshow(A_std)
            # for i in range(self.S):
            #     for j in range(self.S):
            #         text = ax.text(j, i, A_std[i, j],
            #                        ha="center", va="center", color="w")
            # fig.tight_layout()
            # plt.show()
            # plt.imshow(A_std)
            # plt.hist(A_std)

            lst_A = []
            for i in range(self.n):
                same = np.random.normal(1, self.var_qx, size=(self.S, self.S))
                lst_A.append(np.multiply(same, A_std))
                # lst_A.append(A_std)
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
                same = np.random.normal(1, self.var_qx, size=(self.S, self.S))
                lst_A.append(np.multiply(same, A_std))
                # lst_A.append(A_std)

            self.A = sp.linalg.block_diag(*lst_A)

    def _spawn_D(self):
        """
        spawn the matrix D
        """
        temp = np.concatenate([self.d * np.identity(self.S)] * self.n, axis=1)
        self.D = np.concatenate([temp] * self.n, axis=0)

    def ode(self, t=0, N=0):
        """
        the non-dispersal LV equation represented with matrix
        :param N: current number matrix of species
        :return: dN / dt
        """

        temp1 = np.multiply(N, self.M + np.matmul(self.A, N))
        temp2 = np.matmul(self.D, N)
        P = np.ones(self.n*self.S)
        temp3 = np.multiply(N, np.matmul(self.D, P))
        return temp1 + temp2 - temp3

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
        Find the fixed point.Record species numbers N in every 2 steps.
        Stop computing if iteration exceeds maxIteration.
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

            if iteration % 1000 == 0:
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
            # Stop if iteration time exceeds the maximum
            if iteration > self.maxIter:
                # N_new = None
                self.unstableReason = 'iteration overflow'
                break

        self.Nf = N_new

    def calc_jacobian(self, N_fixed: None | np.ndarray = None):
        """
        calculate Jacobian matrix
        :return:
        """
        if N_fixed is None:
            Nf = self.Nf
        else:
            Nf = N_fixed

        N_diag = np.diag(Nf.reshape(self.n*self.S))
        temp1 = np.matmul(N_diag, self.A)
        P = np.ones(self.n * self.S)
        temp2 = np.diag(self.M + np.matmul(self.A, Nf) - np.matmul(self.D, P))
        temp3 = self.D
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

    def compute(self, N_fixed: None | np.ndarray = None):
        """
        compute the fixed point
        :return:
        """
        if N_fixed is None:
            self.spawn()
            self.findFixed()
            self.calc_jacobian()
            self.calc_eigenvalue()
        else:
            self.spawn()
            self.calc_jacobian(N_fixed)
            self.calc_eigenvalue()

        # RK45 module, but have some problem
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

        # theoretical fixed point
        # Xf = np.linalg.solve(self.A, -self.M)
        # print(f'xf:{Xf}')

    def ploteigval(self, ax: plt.Axes):
        """
        plot the eigenvalues of Jacobian matrix on the complex plane
        :return:
        """
        x = self.eigval.real
        x_max = max(x)
        y = self.eigval.imag

        ax.scatter(x, y)
        ax.set_title('Eigenvalues of Jacobian matrix')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.axvline(x=x_max, ls='--', c='r')

    def plotN(self, ax: plt.Axes):
        """
        plot N vs. iterations
        :return:
        """
        ax.plot(self.xlst, np.concatenate(self.N_lst, axis=0))
        ax.set_title(f'Number of species vs. iteration times')
        ax.set_xlabel('iteration')
        ax.set_ylabel('N')

    def plotParam(self, ax: plt.Axes):
        """
        Display the network parameters.
        :return:
        """
        params = f'''
            {NetworkParameter.randomSeed.value}: {self.seed}
            {NetworkParameter.species.value}: {self.S}
            {NetworkParameter.patches.value}:{self.n}
            {NetworkParameter.dispersal.value}: {self.d}
            {NetworkParameter.connectance.value}: {self.c}
            {NetworkParameter.var_alpha.value}: {self.var_alpha}
            {NetworkParameter.var_qx.value}: {self.var_qx}\n
            {NetworkParameter.initial.value}: {self.initial}'''

        if self.initial == 'random':
            params += f'''
            {NetworkParameter.m.value}: {self.m}
            {NetworkParameter.n0.value}: {self.n0}
            '''
        elif self.initial == 'fixed':
            params += f'''
            M = - AN
            N0 ~ N(1, {self.var_n0})
            '''

        ax.set_title('Parameter List')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.annotate(params, (-10, -2), textcoords='offset points', fontsize=11)

    def plotboth(self):
        """
        plot: N, eigenvalues, parameters
        :return:
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 6),
                                gridspec_kw={'width_ratios': [1.5, 1.5, 0.8], 'left': 0.07, 'right': 0.96})
        plt.subplots_adjust(wspace=0.3)
        self.plotN(axs[0])
        self.ploteigval(axs[1])
        self.plotParam(axs[2])
        fig.suptitle(f'Changing Parameter: {self.change[0].value} = {self.change[1]}')
        plt.show()


class NetworkManager:
    """
    This class works as an interface to compute networks.
    """
    def __init__(self, species=1000, patches=1, randomSeed=1, m=1, n0=1.1, initial='fixed', var_n0 = 0.1,
                 dispersal=0, dt=1e-2, maxIteration=2e3, maxError=1e-4,
                 connectance=0.1, var_alpha=0.01, var_qx=0, change=NetworkParameter.randomSeed):
        """
        The parameters of the network.
        :param species: number of species
        :param patches: number of patches
        :param randomSeed: seed of random numbers
        :param m: intraspecific interaction strength
        :param n0: initial N
        :param initial: initial mode.
            'random'--randomly spawn A and M; 'fixed'--spawn M depending on A to get fixed point (1,1,...,1)T
        :param var_n0: the variance of N0
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
                       'm': m,
                       'n0': n0,
                       'initial': initial,
                       'var_n0': var_n0,
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
            self.config[param.value] = var
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

    def knownNf(self):
        """
        Compute the Jacobian matrix with known Nf=(1,1,...,1)^T
        :return:
        """
        Nf = np.ones(self.config['species'] * self.config['patches'])
        net = DispersalNetwork(self.config)
        net.compute(Nf)
        fig, axs = plt.subplots()
        net.ploteigval(axs)
        plt.show()


if __name__ == '__main__':
    net_manager = NetworkManager()
    # net_manager.changeParam(NetworkParameter.randomSeed, 1, 101, 1)
    # net_manager.changeParam(NetworkParameter.dispersal, 50, 1.1, 0.1)
    # net_manager.changeParam(NetworkParameter.connectance, 0.63, 0.64, 0.0001)
    # net_manager.changeParam(NetworkParameter.var_alpha, 1.6, 2, 0.1)
    # net_manager.changeParam(NetworkParameter.var_qx, 0, 0.6, 0.1)
    net_manager.computeNet()
    # net_manager.knownNf()
