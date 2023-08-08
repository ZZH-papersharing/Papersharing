import numpy as np
import scipy as sp
from enum import Enum
from matplotlib import pyplot as plt


class NetParam(Enum):
    """
    Define the name of the network parameters.
    """
    species = r'species: $S=$'  # number of species
    patches = r'patches: $n=$'  # number of patches
    randomSeed = 'randomSeed'  # seed of random numbers

    initial = 'initial mode'  # initial mode
    growth = r'growth rate: $m=$'  # intra-specific interaction strength
    N0 = 'N0'
    n0 = 'initial N'  # initial N
    miu_n0 = 'miu_n0'
    sgm_n0 = 'sgm_n0'  # the variance of N0
    connectance = r'connectance: $c=$'  # Proportion of realized interactions among all possible ones
    sgm_aij = r'$\sigma _{a }=$'  # the variance of the distribution N(0, var) of a_ij
    sgm_qx = r'$\sigma _{q }$='  # the variance of the distribution N(1, var) of q_ij
    Alpha0 = 'Alpha0'
    alpha0 = 'alpha0'  # initial alpha
    miu_alpha = 'miu_alpha'
    sgm_alpha = 'sgm_alpha'
    Adiag = 'intraspecific interaction'
    dispersal = r'dispersal: $d=$'  # dispersal rate
    kappa = r'$\kappa=$'

    rho = r'correlation: $\rho=$'  # correlation between patches
    n_e = r'$n_{e}=$'  # the effective number of ecologically independent patches in the meta-ecosystem
    left1 = r'$\sigma \sqrt{c \left (S-1 \right ) } =$'  # the left term of May's inequality
    left2 = r'$\sigma \sqrt{c \left (S-1 \right ) /n_{e}  } =$'  # the left term of May's inequality

    method = 'computation method'  # numerical computation method
    dt = 'dt'  # time interval
    maxIteration = 'maxIteration'  # max iteration time
    maxError = 'maxError'  # max error in iteration


class DirectionalPreferenceNetwork:
    def __init__(self, config: dict):
        self.seed = config[NetParam.randomSeed]
        np.random.seed(self.seed)

        self.S = config[NetParam.species]
        self.n = config[NetParam.patches]
        self.m = config[NetParam.growth]
        self.c = config[NetParam.connectance]
        self.sgm_aij = config[NetParam.sgm_aij]
        self.sgm_qx = config[NetParam.sgm_qx]
        self.Adiag = config[NetParam.Adiag]

        self.initial = config[NetParam.initial]
        self.growth = config[NetParam.growth]
        self.n0 = config[NetParam.n0]
        self.miu_n0 = config[NetParam.miu_n0]
        self.sgm_n0 = config[NetParam.sgm_n0]
        self.alpha0 = config[NetParam.alpha0]
        self.miu_alpha = config[NetParam.miu_alpha]
        self.sgm_alpha = config[NetParam.sgm_alpha]

        self.d = config[NetParam.dispersal]
        self.kappa = config[NetParam.kappa]

        self.method = config[NetParam.method]
        self.dt = config[NetParam.dt]  # time interval
        self.maxIter = config[NetParam.maxIteration]  # max iteration time
        self.maxError = config[NetParam.maxError]  # max error in iteration
        # self.change = [config['change'], config[config['change'].value]]  # the varying parameter [name, value]

        self.M, self.A, self.D, self.K = None, None, None, None
        self.P2 = None
        self.N0, self.Alpha0 = config.get(NetParam.N0, None), config.get(NetParam.Alpha0, None)
        self.N_f, self.Alpha_f = None, None
        self.flow, self.absflow, self.entropy, self.N_var = [], [], [], []

        self.iter_lst, self.N_lst, self.Alpha_lst = [], [], []

    def spawn(self):
        """
        spawn A, M, N0\n
        A: normal distribution N(0, 1)\n
        M: (1, 1, ..., 1)
        N0: (1, 1, ..., 1)
        :return:
        """
        self._spawn_A(mode=1)

        # use 1-D matrix
        self.miu_alpha = 1 / self.n
        if self.initial == 'fixed':
            # let (1,1,...,1)T as the fixed point in non-dispersal situation
            self.M = np.matmul(self.A, -1 * np.ones(self.S * self.n))
            if self.N0 is None:
                self.N0 = np.random.normal(self.miu_n0, self.sgm_n0, size=self.S * self.n)
            if self.Alpha0 is None:
                self.Alpha0 = np.random.normal(self.miu_alpha, self.sgm_alpha, size=self.S * self.n * self.n)

        elif self.initial == 'random':
            self.M = self.m * np.ones(self.S * self.n)
            if self.N0 is None:
                self.N0 = self.n0 * np.ones(self.S * self.n)
            if self.Alpha0 is None:
                self.Alpha0 = self.alpha0 * np.ones(self.S * self.n * self.n)

        # varify the sum of each S alpha's
        # flag = True
        # for p in range(0, self.S*self.n*self.n, self.S*self.n):
        #     for i in range(p, p+self.S):
        #         # if self.Alpha0[i+self.S*(self.n-1)] != 1 - sum(self.Alpha0[i: i+self.S*(self.n-1): self.S]):
        #         #     flag = False
        #         if sum(self.Alpha0[i: i+self.S*self.n: self.S]) != 1:
        #             flag = False
        # print(flag)

        self.D = self.d * np.ones(self.S * self.n)
        self.K = self.kappa * np.ones(self.S * self.n * self.n)

        temp = np.concatenate([np.identity(self.S)] * self.n, axis=1)
        self.P2 = sp.linalg.block_diag(*[temp] * self.n)

    def _spawn_A(self, mode=1):
        """
        spawn the matrix A
        :param mode: 1--random web; 2--food web (might have some problem)
        :return:
        """

        if mode == 1:
            self.rho = 1 / np.sqrt(1 + (self.sgm_qx / self.sgm_aij) ** 2)
            self.n_e = self.n / (1 + (self.n - 1) * self.rho)
            self.left1 = self.sgm_aij * np.sqrt(self.c * (self.S - 1))
            self.left2 = self.sgm_aij * np.sqrt(self.c * (self.S - 1) / self.n_e)
            # print(f'left1: {self.left1}')

            A_std = np.random.normal(0, self.sgm_aij, size=(self.S, self.S))
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

        # food web, but wasn't updated for versions
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

    def ode(self, t=0, N=0, Alpha=0):
        """
        the preference-dispersal LV equation represented with matrix
        :param N: current number matrix of species
        :return: dN / dt
        """
        # Compute N
        G = self.M + np.matmul(self.A, N)
        term_N_1 = np.multiply(N, G)

        N_patch_lst = [N[i: i + self.S] for i in range(0, self.n * self.S, self.S)]
        Ndiag = [np.diag(np.concatenate([v] * self.n)) for v in N_patch_lst]
        P1 = np.concatenate(Ndiag, axis=1)

        term_N_2 = np.matmul(P1, Alpha)
        term_N_3 = np.multiply(N, np.matmul(self.P2, Alpha))
        result1 = term_N_1 + np.multiply(self.D, (term_N_2 - term_N_3))

        # Compute Alpha
        term_Afa_1 = np.multiply(self.K, Alpha)
        P3 = np.concatenate([G] * self.n)

        G_patch_lst = [G[i: i + self.S] for i in range(0, self.n * self.S, self.S)]
        Gdiag_row = np.concatenate([np.diag(v) for v in G_patch_lst], axis=1)
        Gdiag = np.concatenate([Gdiag_row] * self.n, axis=0)
        P4 = sp.linalg.block_diag(*[Gdiag] * self.n)

        term_Afa_2 = P3 - np.matmul(P4, Alpha)
        result2 = np.multiply(term_Afa_1, term_Afa_2)

        return result1, result2

    def ode_spRK4(self, t=0, y0: np.ndarray = None):
        N, Alpha = y0[: self.S * self.n], y0[self.S * self.n:]
        return np.concatenate(self.ode(N=N, Alpha=Alpha))

    def RK4(self, N, Alpha, dt):
        """
        4th-order Runge-Kutta method
        :param N: current number matrix of species
        :param dt: time interval
        :return: N_n+1
        """
        k11, k12 = self.ode(N=N, Alpha=Alpha)
        k21, k22 = self.ode(N=N + k11 * dt / 2, Alpha=Alpha + k12 * dt / 2)
        k31, k32 = self.ode(N=N + k21 * dt / 2, Alpha=Alpha + k22 * dt / 2)
        k41, k42 = self.ode(N=N + k31 * dt, Alpha=Alpha + k32 * dt)

        result1 = N + dt * (k11 + 2 * k21 + 2 * k31 + k41) / 6
        result2 = Alpha + dt * (k12 + 2 * k22 + 2 * k32 + k42) / 6

        return result1, result2

    def findFixed(self, mode=1):
        """
        Find the fixed point.Record species numbers N in every 2 steps.
        Stop computing if iteration exceeds maxIteration.
        :return:
        """
        if mode == 1:
            N_old = self.N0
            N_new = N_old
            Alpha_old = self.Alpha0
            Alpha_new = Alpha_old
            # if all elements in |N_new - N_old| are < maxError, regard the point as a fixed point
            iteration = 0
            index = range(0, self.S * self.n, self.S)
            while True:  # sum(abs(N_new - N_old) > self.maxError) > 0:
                if iteration % 2 == 0:
                    self.iter_lst.append(iteration)
                    # self.N_lst.append(N_new[index].reshape(1, -1))
                    # self.Alpha_lst.append(Alpha_new[index].reshape(1, -1))
                    self.N_lst.append(N_new.reshape(1, -1))
                    self.Alpha_lst.append(Alpha_new.reshape(1, -1))

                if iteration % 1000 == 0:
                    print(iteration)

                temp1, temp2 = N_new, Alpha_new
                N_new, Alpha_new = self.RK4(N_old, Alpha_old, self.dt)
                N_old, Alpha_old = temp1, temp2
                iteration += 1

                # # if some Ni <= 0, the fixed point does not exist
                # if sum(N_new <= 0) > 0 | sum(Alpha_new <= 0) > 0:
                #     # N_new = None
                #     # self.unstableReason = 'some specie(s) extinguish'
                #     break
                # Stop if iteration time exceeds the maximum
                if iteration > self.maxIter:
                    # N_new = None
                    # self.unstableReason = 'iteration overflow'
                    break

            self.N_f, self.Alpha_f = N_new, Alpha_new

        elif mode == 2:
            solver = sp.integrate.RK45(self.ode_spRK4, t0=0, y0=np.concatenate([self.N0, self.Alpha0]),
                                       t_bound=self.dt * self.maxIter, max_step=10 * self.dt)
            for i in range(int(self.maxIter)):
                if i % 2 == 0:
                    self.iter_lst.append(solver.t / self.dt)
                    self.N_lst.append(solver.y[: self.S * self.n].reshape(1, -1))
                    self.Alpha_lst.append(solver.y[self.S * self.n:].reshape(1, -1))

                if i % 1000 == 0:
                    print(i)

                solver.step()

                if solver.status == 'finished':
                    break

            self.N_f, self.Alpha_f = solver.y[: self.S * self.n], solver.y[self.S * self.n:]

    def analysis(self):
        """
        compute the flow of each species
        :return:
        """
        for i in range(self.S):
            Ni_pch = self.N_f[range(i, self.n * self.S, self.S)]
            Ni_spc = np.concatenate([Ni_pch.reshape(-1, 1)] * self.n, axis=1)
            Alpha_spc = self.Alpha_f[range(i, self.n * self.n * self.S, self.S)].reshape(self.n, self.n)

            single_flow = np.multiply(Ni_spc, Alpha_spc)
            self.flow.append(self.d * (single_flow - single_flow.T))

            entropy_spc = sum(sum(np.multiply(-Alpha_spc, np.log(Alpha_spc)))) / self.n
            self.entropy.append(entropy_spc)
        # self.absflow = [sum(sum(abs(f))) for f in self.flow]
        self.absflow = list(map(lambda x: sum(sum(abs(x))), self.flow))

        self.N_var = [np.var(self.N_f[i:i + self.S]) for i in range(0, self.n * self.S, self.S)]

    def compute(self):
        """
        compute the fixed point
        :return:
        """
        self.spawn()
        self.findFixed(mode=self.method)
        self.analysis()

    def disp(self):
        """
        display network information
        :return:
        """
        print(f'Num. Species:{self.S}')
        print(f'Seed: {self.seed}')
        print(f'N_f: {self.N_f}')
        print(f'Alpha_f: {self.Alpha_f}')

    def plotParam(self):
        """
        Display the network parameters.
        :return:
        """
        params = f'''
            {NetParam.randomSeed.value}: {self.seed}
            {NetParam.species.value}{self.S}
            {NetParam.patches.value}{self.n}
            {NetParam.dispersal.value}{self.d}
            {NetParam.kappa.value}{self.kappa}
            {NetParam.connectance.value}{self.c}
            {NetParam.sgm_aij.value}{self.sgm_aij}
            {NetParam.sgm_qx.value}{self.sgm_qx}
            {NetParam.rho.value}{round(self.rho, 2)}
            {NetParam.n_e.value}{round(self.n_e, 2)}
            {NetParam.left1.value}{round(self.left1, 2)}
            {NetParam.left2.value}{round(self.left2, 2)}\n
            {NetParam.initial.value}: {self.initial}'''

        if self.initial == 'random':
            params += f'''
            {NetParam.growth.value}{self.growth}
            {NetParam.n0.value}: {self.n0}
            '''
        elif self.initial == 'fixed':
            params += rf'''
            $M = - AN$
            $N_{0} \sim N({self.miu_n0}, {self.sgm_n0} ^{2})$
            $\alpha_{0} \sim N({self.miu_alpha}, {self.sgm_alpha} ^{2})$
            '''

        params += f'''
            {NetParam.Adiag.value}:\n\t''' + \
                  r'        $m_{intra}=$' + \
                  f'{self.Adiag}'

        method = {1: "personal RK45", 2: "scipy.integrate.RK45"}[self.method]
        params += f'''
            {NetParam.method.value}:
                {method}'''

        fig: plt.Figure = plt.figure(figsize=(3, 6))
        ax: plt.Axes = fig.subplots()
        ax.set_title('Parameter List')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-6, 0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.annotate(params, (-10, -5.5), textcoords='offset points', fontsize=10)

    def plotNandAlpha(self):
        fig: plt.Figure = plt.figure(figsize=(15, 6))
        axs: list[plt.Axes] = fig.subplots(1, 2,
                                           gridspec_kw={'width_ratios': [1.5, 1.5], 'left': 0.07, 'right': 0.96})
        fig.subplots_adjust(wspace=0.3)
        fig.suptitle(r'N and $\alpha$ vs. iteration times')

        axs[0].plot(self.iter_lst, np.concatenate(self.N_lst, axis=0))
        axs[0].set_title(f'Number of species vs. iteration times')
        axs[0].set_xlabel('iteration')
        axs[0].set_ylabel('N')

        axs[1].plot(self.iter_lst, np.concatenate(self.Alpha_lst, axis=0))
        axs[1].set_title(r'$\alpha$ vs. iteration times')
        axs[1].set_xlabel('iteration')
        axs[1].set_ylabel(r'$\alpha$')

    def plotHist_N(self):
        """
        Plot the histogram of $N_{0}, N_{f}$
        :return:
        """
        fig: plt.Figure = plt.figure()
        axs: list[plt.Axes] = fig.subplots(1, 2)
        fig.suptitle(r'histogram of $N_{0}, N_{f}$')
        # index = range(0, self.S * self.n, self.S)
        minN = min(np.min(self.N0), np.min(self.N_f))
        maxN = max(np.max(self.N0), np.max(self.N_f))
        ran = (minN * 0.9, maxN * 1.1)
        _ = axs[0].hist(self.N0, range=ran, bins=20, weights=None)
        _ = axs[1].hist(self.N_f, range=ran, bins=20, weights=None)
        axs[0].set_xlabel(r'$N_{0}$')
        axs[1].set_xlabel(r'$N_{f}$')
        axs[0].set_ylabel('Count')
        axs[1].set_ylabel('Count')

    def plotHist_Afa(self):
        """
        Plot the histogram of $\alpha_{0}, \alpha_{f}$
        :return:
        """
        fig: plt.Figure = plt.figure()
        axs: list[plt.Axes] = fig.subplots(1, 2)
        fig.suptitle(r'histogram of $\alpha_{0}, \alpha_{f}$')
        # index = range(0, self.S * self.n, self.S)
        _ = axs[0].hist(self.Alpha0, range=(0, 1), bins=30, weights=None)
        _ = axs[1].hist(self.Alpha_f, range=(0, 1), bins=30, weights=None)
        axs[0].set_xlabel(r'$\alpha_{0}$')
        axs[1].set_xlabel(r'$\alpha_{f}$')
        axs[0].set_ylabel('Count')
        axs[1].set_ylabel('Count')

    def plotFlow(self, idx: int):
        """
        Plot the flow of species i from patch k to patch l
        :return:
        """
        to_plot = self.flow[idx]
        fig: plt.Figure = plt.figure(figsize=(9, 5))
        ax: plt.Axes = fig.subplots()
        im = ax.imshow(to_plot)
        cbar = ax.figure.colorbar(im, ax=ax)
        for k in range(self.n):
            for l in range(self.n):
                text = ax.text(l, k, round(to_plot[k, l], 2),
                                ha="center", va="center", color="w")
        # fig.tight_layout()
        ax.set_xlabel('Arrival Patch')
        ax.set_ylabel('Departure Patch')
        ax.set_title(f'Flow of species #{idx}')

    def plotAll(self, histN=True, histAfa=True, flow=True):
        """
        plot: N, eigenvalues, parameters
        :return:
        """
        self.plotParam()
        self.plotNandAlpha()
        if histN:
            self.plotHist_N()
        if histAfa:
            self.plotHist_Afa()
        if flow:
            for i in range(self.S):
                self.plotFlow(idx=i)
                plt.show()


class NetworkManager:
    """
    This class works as an interface to compute networks.
    """

    def __init__(self, species=10, patches=5, randomSeed=1,
                 initial='random', growth=1, n0=1, alpha0=0.2, N0=None, Alpha0=None,
                 miu_n0=1, sgm_n0=0.05, miu_alpha=0.5, sgm_alpha=0.01,
                 connectance=0.3, sgm_aij=0.1, sgm_qx=0.1, Adiag=1, dispersal=10, kappa=0.1,
                 method=2, dt=1e-2, maxIteration=10e3, maxError=1e-4,
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
                       NetParam.Adiag: Adiag,
                       NetParam.dispersal: dispersal,
                       NetParam.kappa: kappa,

                       NetParam.method: method,
                       NetParam.dt: dt,
                       NetParam.maxIteration: maxIteration,
                       NetParam.maxError: maxError,
                       }

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
        net = DirectionalPreferenceNetwork(self.config)
        net.compute()
        # if not net.stable:
        #     continue
        net.disp()
        net.plotAll()

    def plotaxes(self, X, Y, ax: plt.Axes, title='', xlabel='', ylabel='',
                 xlim: None | tuple = None, ylim: None | tuple = None):
        ax.plot(X, Y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

    def change(self, x_var: NetParam, start, end, step):
        """
        plot flow vs. kappa
        :return:
        """
        x_lst = np.arange(start, end, step)
        flow_lst = []
        entropy_lst = []
        Nvar_lst = []
        net = None
        for value in x_lst:
            self.config[x_var] = value
            net = DirectionalPreferenceNetwork(self.config)
            net.compute()
            flow_lst.append(np.array(net.absflow).reshape(1, -1))
            entropy_lst.append(np.array(net.entropy).reshape(1, -1))
            Nvar_lst.append(np.array(net.N_var).reshape(1, -1))

        fig, axs = plt.subplots(1, 3)
        self.plotaxes(X=x_lst, Y=np.concatenate(flow_lst, axis=0), ax=axs[0],
                      title=f'absolute flow of each species', xlabel=f'{x_var.value}', ylabel='absolute flow')
        self.plotaxes(X=x_lst, Y=np.concatenate(entropy_lst, axis=0), ax=axs[1],
                      title=f'entropy of each species', xlabel=f'{x_var.value}', ylabel='entropy')
        self.plotaxes(X=x_lst, Y=np.concatenate(Nvar_lst, axis=0), ax=axs[2],
                      title=f'variance of N in each patches', xlabel=f'{x_var.value}', ylabel='variance')

        net.plotParam()
        plt.show()

    def origin(self, mode=1, runtime=5):
        S, n = self.config[NetParam.species], self.config[NetParam.patches]
        xN, xAfa, N0_lst, Alpha0_lst = [], [], [], []

        np.random.seed(1)
        labels = []
        if mode == 1:
            xN, xAfa, N0_lst, Alpha0_lst = self.initial_lst(S, n, run=runtime, mode=mode)
            labels = ['runs', 'runs']
        elif mode == 2:
            xN, xAfa, N0_lst, Alpha0_lst = self.initial_lst(S, n, run=runtime, mode=mode)
            labels = [r'$\sigma_{N_{0}}$', r'$\sigma_{\alpha_{0}}$']
        elif mode == 3:
            xN, xAfa, N0_lst, Alpha0_lst = self.initial_lst(S, n, run=runtime, mode=mode)
            labels = [r'$\mu_{N_{0}}$', r'$\mu_{\alpha_{0}}$']
        elif mode == 4:
            xN, xAfa, N0_lst, Alpha0_lst = self.initial_lst(S, n, run=runtime, mode=mode)
            labels = [r'$\mu_{\alpha_{0}}$', r'$\mu_{\alpha_{0}}$']

        fig1: plt.Figure = plt.figure()
        axs: list[list[plt.Axes]] = fig1.subplots(2, 2)
        fig1.subplots_adjust(wspace=0.3, hspace=0.3)

        net = None
        N1_lst, Alpha1_lst = [], []
        for Alpha0 in Alpha0_lst:
            self.config[NetParam.N0] = None
            self.config[NetParam.Alpha0] = Alpha0
            net = DirectionalPreferenceNetwork(self.config)
            net.compute()
            # net.plotAll()
            # plt.show()
            N1_lst.append(net.N_f.reshape(1, -1))
            Alpha1_lst.append(net.Alpha_f.reshape(1, -1))
            # N1_lst.append(net.N_f[0])
            # Alpha1_lst.append(net.Alpha_f[0])

        self.plotaxes(X=xAfa, Y=np.concatenate(N1_lst, axis=0), ax=axs[1][0],
                      title=r'$N_{f}$ affected by ' + labels[1], xlabel=labels[1], ylabel=r'$N_{f}$',
                      ylim=(np.min(N1_lst) * 0.8, np.max(N1_lst) * 1.1))
        self.plotaxes(X=xAfa, Y=np.concatenate(Alpha1_lst, axis=0), ax=axs[1][1],
                      title=r'$\alpha_{f}$ affected by ' + labels[1], xlabel=labels[1], ylabel=r'$\alpha_{f}$',
                      ylim=(np.min(Alpha1_lst) * 0.8, np.max(Alpha1_lst) * 1.1))

        net.plotAll(histN=False, histAfa=False, flow=False)

        plt.show()

    def initial_lst(self, S, n, run, mode=1):
        if mode == 1:
            runs = range(run)
            mu_n, sgm_n = self.config[NetParam.miu_n0], self.config[NetParam.sgm_n0]
            mu_afa, sgm_afa = 1 / n, self.config[NetParam.sgm_alpha]
            N0_lst = [np.random.normal(mu_n, sgm_n, size=S * n) for i in runs]
            Alpha0_lst = [np.random.normal(mu_afa, sgm_afa, size=S * n * n) for i in runs]
            return runs, runs, N0_lst, Alpha0_lst

        elif mode == 2:
            sgmN = np.linspace(0, 0.3, run)
            sgmAfa = np.linspace(0, 0.03, run)
            mu_n, mu_afa = self.config[NetParam.miu_n0], 1 / n
            N0_lst = [np.random.normal(mu_n, sgm_n, size=S * n) for sgm_n in sgmN]
            Alpha0_lst = [np.random.normal(mu_afa, sgm_afa, size=S * n * n) for sgm_afa in sgmAfa]
            return sgmN, sgmAfa, N0_lst, Alpha0_lst

        elif mode == 3:
            muN = np.linspace(1, 100, run)
            sgm_n = self.config[NetParam.sgm_n0]
            N0_lst = [np.random.normal(mu_n, sgm_n, size=S * n) for mu_n in muN]
            return muN, [], N0_lst, []

        elif mode == 4:
            muA = np.linspace(0.1, 1 / n, run)
            Alpha0_lst = [np.random.normal(mu_afa, 0.01, size=S * n * n) for mu_afa in muA]
            return [], muA, [], Alpha0_lst

    def origin_alpha(self, mode=2, runtime=10):
        S, n = self.config[NetParam.species], self.config[NetParam.patches]
        np.random.seed(1)
        xAfa, Alpha0_lst = range(runtime), self.initial_alpha(S, n, runtime, mode)
        fig: plt.Figure = plt.figure()
        axs: list[plt.Axes] = fig.subplots(1, 2)

        models = {1: 'interval', 2: 'proportion', 3: 'dirichlet'}
        fig.suptitle(r'First group of $\alpha:$'
                     + r'$\left ( \alpha _{1}^{11} ,\alpha _{1}^{12},\dots \alpha _{1}^{1n} \right )$'
                     + f'generation method: {models[mode]}')

        index = range(0, S * n, S)
        print(Alpha0_lst[0][index])
        # _ = axs[0].hist(Alpha0_lst[0][index], range=(0, 1), bins=30, weights=None)
        # plt.show()
        # temp = [Alpha0_lst[i][index].reshape(1, -1) for i in range(runtime)]
        # self.plotaxes(X=xAfa, Y=np.concatenate(temp, axis=0), ax=axs[0],
        #               xlabel='runs', ylabel=r'$\alpha_{0}$', ylim=(0, 1))

        net = None
        Alpha1_lst = []
        for Alpha0 in Alpha0_lst:
            self.config[NetParam.N0] = None
            self.config[NetParam.Alpha0] = Alpha0
            # _ = axs[0].hist(Alpha0_lst[0], range=(0, 1), bins=30, weights=None)
            net = DirectionalPreferenceNetwork(self.config)
            net.compute()
            # net.plotNandAlpha()
            # plt.show()
            Alpha1_lst.append(net.Alpha_f[index].reshape(1, -1))
            # N1_lst.append(net.N_f[0])
            # Alpha1_lst.append(net.Alpha_f[0])

        self.plotaxes(X=xAfa, Y=np.concatenate(Alpha1_lst, axis=0), ax=axs[1],
                      xlabel='runs', ylabel=r'$\alpha_{f}$', ylim=(0, 1))

        net.plotAll(histN=False, histAfa=False, flow=False)

        plt.show()

    def initial_alpha(self, S, n, run, mode=1):
        Alpha0_lst = []
        for _ in range(run):
            alpha_lst = np.empty(shape=(n, n, S))
            for i in range(n):
                rannums = np.empty(shape=(S, n))
                if mode == 1:
                    for j in range(S):
                        values = [0.0, 1.0] + [np.random.random() for _ in range(n - 1)]
                        values.sort()
                        rannums[j] = sorted([values[i + 1] - values[i] for i in range(n)])

                elif mode == 2:
                    for j in range(S):
                        # values = np.sort(np.random.random(size=n))
                        values = np.random.random(size=n)
                        rannums[j] = values / np.sum(values)

                elif mode == 3:
                    afa_drcl = [1] * n
                    # afa_drcl[-1] = 10
                    rannums = np.sort(np.random.dirichlet(afa_drcl, S))

                alpha_lst[i] = rannums.T

            Alpha0_lst.append(alpha_lst.flatten())
        return Alpha0_lst


if __name__ == '__main__':
    net_manager = NetworkManager()
    # net_manager.computeNet()
    # net_manager.change(NetParam.n0, 1, 12, 5)
    net_manager.origin(mode=1)
    # net_manager.origin_alpha()
