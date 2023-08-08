import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from enum import Enum


class NetworkParameter(Enum):
    """
    Define the name of the network parameters.
    """
    species = r'species: $S=$'  # number of species
    patches = r'patches: $n=$'  # number of patches
    randomSeed = 'randomSeed'  # seed of random numbers
    m = r'growth rate: $m=$'  # intra-specific interaction strength
    n0 = 'initial N'  # initial N
    initial = 'initial mode'  # initial mode
    sgm_n0 = 'sgm_n0'  # the variance of N0
    dispersal = r'dispersal: $d=$'  # dispersal rate
    Adiag = 'intraspecific interaction'

    method = 'computation method'  # numerical computation method
    dt = 'dt'  # time interval
    maxIteration = 'maxIteration'  # max iteration time
    maxError = 'maxError'  # max error in iteration
    connectance = r'connectance: $c=$'  # Proportion of realized interactions among all possible ones
    sgm_alpha = r'$\sigma _{\alpha }=$'  # the variance of the distribution N(0, var) of alpha_ij
    sgm_qx = r'$\sigma _{q }$='  # the variance of the distribution N(1, var) of q_ij
    rho = r'correlation: $\rho=$'  # correlation between patches
    n_e = r'$n_{e}=$'  # the effective number of ecologically independent patches in the meta-ecosystem
    left1 = r'$\sigma \sqrt{c \left (S-1 \right ) } =$'  # the left term of May's inequality
    left2 = r'$\sigma \sqrt{c \left (S-1 \right ) /n_{e}  } =$'  # the left term of May's inequality


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
        self.miu_n0 = config['miu_n0']
        self.sgm_n0 = config['sgm_n0']
        self.sgm_alpha = config['sgm_alpha']
        self.sgm_qx = config['sgm_qx']
        self.d = config['dispersal']  # dispersal rate
        self.Adiag = config['Adiag']

        self.method = config['method']
        self.dt = config['dt']  # time interval
        self.maxIter = config['maxIteration']  # max iteration time
        self.maxError = config['maxError']  # max error in iteration
        self.change = [config['change'], config[config['change'].value]]  # the varying parameter [name, value]

        self.N0 = None  # initial N
        self.M, self.A, self.D = None, None, None
        self.rho, self.n_e, self.left = None, None, None  # correlation
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
            self.N0 = np.random.normal(self.miu_n0, self.sgm_n0, size=self.S * self.n)
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
            self.rho = 1 / np.sqrt(1 + (self.sgm_qx / self.sgm_alpha)**2)
            self.n_e = self.n / (1 + (self.n - 1) * self.rho)
            self.left1 = self.sgm_alpha * np.sqrt(self.c * (self.S - 1))
            self.left2 = self.sgm_alpha * np.sqrt(self.c * (self.S - 1) / self.n_e)
            # print(f'left1: {self.left1}')

            A_std = np.random.normal(0, self.sgm_alpha, size=(self.S, self.S))
            Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            lst_A = []
            for i in range(self.n):
                same = np.random.normal(0, self.sgm_qx, size=(self.S, self.S))
                Ax = (A_std + same) * self.rho
                Ax = np.multiply(Ax, Connect)
                Ax -= np.diag(np.diag(Ax) + self.Adiag)

                # print('mean:', np.mean(np.mean(Ax)))
                # print('var:', np.var(Ax))
                # print(Ax)
                # B1, B2 = Ax, A_std
                # arr = B1.reshape(1, -1)[0]
                # _=plt.hist(arr[arr != 0], range=(-0.5, 0.5), bins=100, weights=None)
                # plt.Axes().set_xlim(xlim=(-0.5, 0.5))
                # plt.show()

                # print('rho:', np.corrcoef(np.concatenate([B1.reshape((1, -1)), B2.reshape((1, -1))], axis=0)))

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

                lst_A.append(Ax)
                # lst_A.append(A_std)
            # print(lst_A[1] - lst_A[0])

            self.A = sp.linalg.block_diag(*lst_A)

        elif mode == 2:
            A_std = -1 * np.identity(self.S)
            for i in range(self.S):
                for j in range(self.S):
                    if i < j:
                        if np.random.random() < self.c:
                            A_std[i, j] = np.random.normal(0, self.sgm_alpha) / 5
                    elif i > j:
                        A_std[i, j] = -A_std[j, i]

            lst_A = []
            for i in range(self.n):
                same = np.random.normal(1, self.sgm_qx, size=(self.S, self.S))
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

    def findFixed(self, mode=2):
        """
        Find the fixed point.Record species numbers N in every 2 steps.
        Stop computing if iteration exceeds maxIteration.
        :return:
        """
        if mode == 1:
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

        elif mode == 2:
            solver = sp.integrate.RK45(self.ode, t0=0, y0=self.N0, t_bound=self.dt*self.maxIter, max_step=10*self.dt)
            for i in range(int(self.maxIter)):
                if i % 2 == 0:
                    self.xlst.append(solver.t / self.dt)
                    self.N_lst.append(solver.y.reshape(1, -1))

                if i % 500 == 0:
                    print(i)

                solver.step()

                if solver.status == 'finished':
                    break

            self.Nf = solver.y

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

    def compute(self):
        """
        compute the fixed point
        :return:
        """

        self.spawn()
        self.findFixed(mode=self.method)
        self.calc_jacobian()
        self.calc_eigenvalue()

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

    def plotFlow(self):
        """
        Plot the flow of species i from patch k to patch l
        :return:
        """
        for i in range(self.S):
            Ni_pch = self.Nf[range(i, self.n*self.S, self.S)]
            Ni_spc = np.concatenate([Ni_pch.reshape(1, -1)]*self.n, axis=0)
            flow = self.d * (Ni_spc.T - Ni_spc)
            flowratio = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))

            to_plot = flow
            fig, ax = plt.subplots(1, 2, figsize=(9, 5))
            ax1 = ax[1]
            im = ax1.imshow(to_plot)
            cbar = ax1.figure.colorbar(im, ax=ax)
            for k in range(self.n):
                for l in range(self.n):
                    text = ax1.text(l, k, round(to_plot[k, l], 2),
                                   ha="center", va="center", color="w")
            # fig.tight_layout()
            plt.xlabel('Arrival Patch')
            plt.ylabel('Departure Patch')
            plt.title(f'Flow of species #{i}')

            self.plotParam(ax[0])

            plt.show()

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
        ax.set_xlim(-20, 5)
        ax.set_ylim(-10, 10)
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
            {NetworkParameter.species.value}{self.S}
            {NetworkParameter.patches.value}{self.n}
            {NetworkParameter.dispersal.value}{self.d}
            {NetworkParameter.connectance.value}{self.c}
            {NetworkParameter.sgm_alpha.value}{self.sgm_alpha}
            {NetworkParameter.sgm_qx.value}{self.sgm_qx}
            {NetworkParameter.rho.value}{round(self.rho, 2)}
            {NetworkParameter.n_e.value}{round(self.n_e, 2)}
            {NetworkParameter.left1.value}{round(self.left1, 2)}
            {NetworkParameter.left2.value}{round(self.left2, 2)}\n
            {NetworkParameter.initial.value}: {self.initial}'''

        if self.initial == 'random':
            params += f'''
            {NetworkParameter.m.value}{self.m}
            {NetworkParameter.n0.value}: {self.n0}
            '''
        elif self.initial == 'fixed':
            params += rf'''
            $M = - AN$
            $N_{0} \sim N({self.miu_n0}, {self.sgm_n0} ^{2})$
            '''

        params += f'''
            {NetworkParameter.Adiag.value}:\n\t''' + \
            r'        $m_{intra}=$' + \
            f'{self.Adiag}'

        method = {1: "personal RK45", 2: "scipy.integrate.RK45"}[self.method]
        params += f'''
            {NetworkParameter.method.value}:
                {method}'''

        ax.set_title('Parameter List')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-6, 0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.annotate(params, (-10, -5.5), textcoords='offset points', fontsize=11)

    def plotboth(self):
        """
        plot: N, eigenvalues, parameters
        :return:
        """
        # self.plotFlow()

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
    def __init__(self, species=100, patches=20, randomSeed=1, m=1, n0=1, initial='fixed', miu_n0=1, sgm_n0=0.05,
                 dispersal=8/19, dt=1e-2, maxIteration=4e3, maxError=1e-4, method=2, Adiag=2,
                 connectance=0.3, sgm_alpha=1, sgm_qx=100, change=NetworkParameter.randomSeed):
        """
        The parameters of the network.
        :param species: number of species
        :param patches: number of patches
        :param randomSeed: seed of random numbers
        :param m: intraspecific interaction strength
        :param n0: initial N
        :param initial: initial mode.
            'random'--randomly spawn A and M; 'fixed'--spawn M depending on A to get fixed point (1,1,...,1)T
        :param sgm_n0: the variance of N0
        :param dispersal: dispersal rate
        :param dt: time interval
        :param maxIteration: max iteration time
        :param maxError: max error in iteration
        :param connectance:  Proportion of realized interactions among all possible ones
        :param sgm_alpha: the variance of the distribution N(0, var) of alpha_ij
        :param sgm_qx: the variance of the distribution N(1, var) of q_ij
        """
        self.config = {'randomSeed': randomSeed,
                       'species': species,
                       'patches': patches,
                       'dispersal': dispersal,
                       'Adiag': Adiag,
                       'm': m,
                       'n0': n0,
                       'initial': initial,
                       'miu_n0': miu_n0,
                       'sgm_n0': sgm_n0,
                       'dt': dt,
                       'maxIteration': maxIteration,
                       'maxError': maxError,
                       'connectance': connectance,
                       'sgm_alpha': sgm_alpha,
                       'sgm_qx': sgm_qx,
                       'method': method,
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
        net.compute()
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
