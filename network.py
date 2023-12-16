from enum import Enum

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def critical(x):
    """
    to determine dt
    :param x:
    :return:
    """
    return 2.3 - np.log(x + np.e - 1) / (x + np.e - 1)


class NetParam(Enum):
    """
    Define the name of the network parameters.
    """
    species = {'latex': r'$S$', 'str': 'S'}  # number of species
    patches = {'latex': r'$n$', 'str': 'n'}  # number of patches
    randomSeed = {'latex': 'randomSeed', 'str': 'seed'}  # seed of random numbers
    weightSeed = {'str': 'weightSeed'}

    initial = {'latex': 'initial mode', 'str': 'initialMode'}  # initial mode
    growth = {'latex': r'$m$', 'str': 'growth'}  # intra-specific interaction strength
    N0 = {'latex': 'N0'}
    n0 = {'latex': 'initial N'}  # initial N
    miu_n0 = {'latex': 'miu_n0'}
    sgm_n0 = {'latex': 'sgm_n0'}  # the variance of N0
    connectance = {'latex': r'$c$',
                   'str': 'c'}  # Proportion of realized interactions among all possible ones
    sgm_aij = {'latex': r'$\sigma _{a }$', 'str': 'sgm'}  # the variance of the distribution N(0, var) of a_ij
    sgm_qx = {'latex': r'$\sigma _{q }$', 'str': 'sgmqx'}  # the variance of the distribution N(1, var) of q_ij
    Alpha0 = {'latex': 'Alpha0'}
    alpha0 = {'latex': 'alpha0'}  # initial alpha
    miu_alpha = {'latex': 'miu_alpha'}
    sgm_alpha = {'latex': 'sgm_alpha'}
    Adiag = {'latex': 'intraspecific interaction'}
    dispersal = {'latex': r'$d$', 'str': 'd'}  # dispersal rate
    kappa = {'latex': r'$\kappa$', 'str': 'kpa'}
    scr = {'str': 'signChangeRate'}
    record = 'record'

    method_alpha: dict = {'name': r'model of $\alpha$', 0: 'Personal', 1: 'Softmax'}

    rho = {'latex': r'$\rho$', 'str': 'rho'}  # correlation between patches
    n_e = {'latex': r'$n_{e}$'}  # the effective number of ecologically independent patches in the meta-ecosystem
    left1 = {'latex': r'$\sigma \sqrt{c \left (S-1 \right ) } $'}  # the left term of May's inequality
    left2 = {'latex': r'$\sigma \sqrt{c \left (S-1 \right ) /n_{e}  } $'}  # the left term of May's inequality

    method_ode = 'computation method'  # numerical computation method
    dt = {'latex': 'dt'}  # time interval
    maxIteration = {'latex': 'maxIteration'}  # max iteration time
    maxError = {'latex': 'maxError'}  # max error in iteration


class RandomNetwork:
    def __init__(self, config: dict):
        self.seed = config[NetParam.randomSeed]
        self.weightSeed = config[NetParam.weightSeed]
        np.random.seed(self.seed)

        self.S = config[NetParam.species]
        self.n = config[NetParam.patches]
        self.m = config[NetParam.growth]
        self.c = config[NetParam.connectance]
        self.sgm_aij = config[NetParam.sgm_aij]
        self.sgm_qx = config[NetParam.sgm_qx]
        self.rho = config[NetParam.rho]
        self.Adiag = config[NetParam.Adiag]

        self.initial = config[NetParam.initial]
        self.growth = config[NetParam.growth]
        self.n0 = config[NetParam.n0]
        self.miu_n0 = config[NetParam.miu_n0]
        self.sgm_n0 = config[NetParam.sgm_n0]
        self.alpha0 = config[NetParam.alpha0]
        self.miu_alpha = config[NetParam.miu_alpha]
        self.sgm_alpha = config[NetParam.sgm_alpha]
        self.scr = config[NetParam.scr]
        self.record = config[NetParam.record]

        self.d = config[NetParam.dispersal]
        self.kappa = config[NetParam.kappa]

        self.method_alpha = config[NetParam.method_alpha]
        self.method_ode = config[NetParam.method_ode]
        self.dt = config[NetParam.dt]  # time interval

        p = max(1, np.log10(self.d) + 1)
        if self.kappa >= critical(self.d):
            p += 1
        self.dt = 1 / 10 ** np.ceil(p)
        print(self.dt)

        self.maxIter = config[NetParam.maxIteration]  # max iteration time
        self.maxError = config[NetParam.maxError]  # max error in iteration
        # self.change = [config['change'], config[config['change'].value]]  # the varying parameter [name, value]

        self.Ax_lst = []
        self.M, self.A, self.D, self.K = None, None, None, None
        self.P2, self.G, self.avgG = None, np.empty(self.S * self.n), np.empty(self.S * self.n * self.n)
        self.N0, self.Alpha0 = config.get(NetParam.N0, None), config.get(NetParam.Alpha0, None)
        self.curN, self.curAlpha = None, None
        self.theoryPoints, self.maxTheoryEigvals, self.maxFactEigvals = [], [], []
        self.errors = []
        self.N_f, self.Alpha_f = np.zeros(self.S * self.n), np.zeros(self.S * self.n * self.n)
        self.J, self.eigval = None, None
        self.maxEigval = None
        self.fixpt = []

        self.alpha_species, self.singleflow, self.flow = [], [], []
        self.entropy, self.N_var = [], []
        self.var_Nf, self.var_Alphaf, self.var_flow, self.absflow = [], [], [], []
        self.pst, self.stable = True, False

        self.iter_lst, self.N_lst, self.Alpha_lst, self.G_lst, self.avgG_lst = [], [], [], [], []
        self.iter = 0
        self.sumslope = np.zeros(self.S * self.n)
        self.slopes = []

        self.spawn()

    def spawn(self):
        """
        spawn A, M, N0\n
        A: normal distribution N(0, 1)\n
        M: (1, 1, ..., 1)
        N0: (1, 1, ..., 1)
        :return:
        """
        self._spawn_A(mode=1)
        # print('A:', self.A)

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
            self.sgm_qx = self.sgm_aij * np.sqrt(1 / self.rho - 1)
            lmd = 1 / np.sqrt(1 + (self.sgm_qx / self.sgm_aij) ** 2)
            self.n_e = self.n / (1 + (self.n - 1) * self.rho)
            self.left1 = self.sgm_aij * np.sqrt(self.c * (self.S - 1))
            self.left2 = self.sgm_aij * np.sqrt(self.c * (self.S - 1) / self.n_e)

            # original version, all randomized
            W_std = np.random.normal(0, self.sgm_aij, size=(self.S, self.S))
            Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            offsets = [np.random.normal(0, self.sgm_qx, size=(self.S, self.S)) for i in range(self.n)]
            W_lst = [(W_std + offset) * lmd for offset in offsets]

            # # fix C change W
            # Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            # # print(Connect)
            # np.random.seed(self.weightSeed)
            # W_std = np.random.normal(0, self.sgm_aij, size=(self.S, self.S))
            # offsets = [np.random.normal(0, self.sgm_qx, size=(self.S, self.S)) for i in range(self.n)]
            # W_lst = [(W_std + offset) * lmd for offset in offsets]

            # # fix C, fix Wstd, change offset
            # Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            # W_std = np.random.normal(0, self.sgm_aij, size=(self.S, self.S))
            # np.random.seed(self.weightSeed)
            # offsets = [np.random.normal(0, self.sgm_qx, size=(self.S, self.S)) for i in range(self.n)]
            # W_lst = [(W_std + offset) * lmd for offset in offsets]

            # # fix C, fix W strength, change Sign
            # # version 1: consider rho
            # Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            # offsets = [np.random.normal(0, self.sgm_qx, size=(self.S, self.S)) for i in range(self.n)]
            # W_std = np.random.normal(0, self.sgm_aij, size=(self.S, self.S))
            # np.random.seed(self.weightSeed)
            # sgn = np.random.binomial(1, 0.5, size=(self.S, self.S)) * 2 - 1
            # W_lst = [(W_std + offset) * lmd * sgn for offset in offsets]

            # # version 2: ignore rho
            # Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            # W_x = [np.random.normal(0, self.sgm_aij, size=(self.S, self.S)) for i in range(self.n)]
            # np.random.seed(self.weightSeed)
            # sgn = np.random.binomial(1, 0.5, size=(self.S, self.S)) * 2 - 1
            # W_lst = [abs(W) * sgn for W in W_x]

            # # fix C, Sign, change absW
            # Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            # sgn = np.random.binomial(1, 0.5, size=(self.S, self.S)) * 2 - 1
            # np.random.seed(self.weightSeed)
            # W_x = [np.random.normal(0, self.sgm_aij, size=(self.S, self.S)) for i in range(self.n)]
            # W_lst = list(map(lambda x: abs(x) * sgn, W_x))

            # # fix C, absW, change Sign step-by-step
            # Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            # absw = self.sgm_aij * np.sqrt(2 / np.pi)
            # absW_lst = [np.random.binomial(1, 0.5, size=(self.S, self.S)) * 2 * absw - absw for i in range(self.n)]
            # np.random.seed(self.weightSeed)
            # sgn = np.random.binomial(1, self.scr, size=(self.S, self.S)) * (-1)
            # W_lst = list(map(lambda x: x * sgn, absW_lst))

            self.Ax_lst = [Connect * Wx for Wx in W_lst]
            self.Ax_lst = [Ax - np.diag(np.diag(Ax) + self.Adiag) for Ax in self.Ax_lst]
            # for i in range(self.n):
            #     offset = np.random.normal(0, self.sgm_qx, size=(self.S, self.S))
            #     Wx = (W_std + offset) * lmd
            #     Ax = np.multiply(Wx, Connect)
            #     Ax -= np.diag(np.diag(Ax) + self.Adiag)
            #
            #     self.Ax_lst.append(Ax)

            self.A = sp.linalg.block_diag(*self.Ax_lst)
            # print(self.A.shape)
            # print(self.Ax_lst[0])

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

    def initcompute(self):
        self.pst, self.stable = True, False
        self.iter_lst, self.N_lst, self.Alpha_lst, self.G_lst, self.avgG_lst = [], [], [], [], []
        self.curN, self.curAlpha = self.N0, self.Alpha0

    def plotParam(self):
        """
        Display the network parameters.
        :return:
        """
        params = f'''
            {NetParam.randomSeed.value['latex']}: {self.seed}
            species: {NetParam.species.value['latex']}={self.S}
            patches: {NetParam.patches.value['latex']}={self.n}
            disperal: {NetParam.dispersal.value['latex']}={self.d}
            kappa: {NetParam.kappa.value['latex']}={self.kappa}
            connectance: {NetParam.connectance.value['latex']}={self.c}
            strength: {NetParam.sgm_aij.value['latex']}={self.sgm_aij}
            correlation: {NetParam.rho.value['latex']}={self.rho}
            {NetParam.n_e.value['latex']}={round(self.n_e, 2)}
            {NetParam.left1.value['latex']}={round(self.left1, 2)}
            {NetParam.left2.value['latex']}={round(self.left2, 2)}\n
            {NetParam.initial.value['latex']}: {self.initial}'''

        if self.initial == 'random':
            params += f'''
            growth rate: {NetParam.growth.value['latex']}={self.growth}
            {NetParam.n0.value['latex']}: {self.n0}
            '''
        elif self.initial == 'fixed':
            params += rf'''
            $M = - AN$
            $N_{0} \sim N({self.miu_n0}, {self.sgm_n0} ^{2})$
            $\alpha_{0} \sim N({self.miu_alpha}, {self.sgm_alpha} ^{2})$
            '''

        params += f'''
            {NetParam.Adiag.value['latex']}:\n\t''' + \
                  r'        $m_{intra}=$' + \
                  f'{self.Adiag}'

        method_ode = {1: "personal RK45", 2: "scipy.integrate.RK45"}[self.method_ode]
        params += f'''
            {NetParam.method_ode.value}:
                {method_ode}'''
        params += f'''
            {NetParam.method_alpha.value['name']}:
                {self.method_alpha}'''

        fig: plt.Figure = plt.figure(figsize=(3, 6.5))
        ax: plt.Axes = fig.subplots()
        ax.set_title('Parameter List')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-6, 0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.annotate(params, (-10, -5.5), textcoords='offset points', fontsize=10)

    def plotNandAlpha(self):
        fig: plt.Figure = plt.figure(figsize=(15, 6))
        axs: list[list[plt.Axes]] = fig.subplots(2, 2,
                                                 gridspec_kw={'left': 0.07, 'right': 0.96})
        fig.subplots_adjust(wspace=0.2, hspace=0.35)
        fig.suptitle(r'N and $\alpha$ vs. iteration times')

        axs[0][0].plot(self.iter_lst, np.concatenate(self.N_lst, axis=0))
        axs[0][0].set_title(f'Number of species vs. iteration times')
        axs[0][0].set_xlabel('iteration')
        axs[0][0].set_ylabel('N')

        axs[0][1].scatter(range(len(self.maxTheoryEigvals)), self.maxTheoryEigvals)
        axs[0][1].set_title(r'maxTheoryEigvals')
        axs[0][1].set_xlabel('')
        axs[0][1].set_ylabel('')

        axs[1][0].plot(self.iter_lst, np.concatenate(self.Alpha_lst, axis=0))
        axs[1][0].set_title(r'$\alpha$ vs. iteration times')
        axs[1][0].set_xlabel('iteration')
        axs[1][0].set_ylabel(r'$\alpha$')

        axs[1][1].plot(self.iter_lst, np.concatenate(self.G_lst, axis=0))
        axs[1][1].set_title(r'G vs. iteration times')
        axs[1][1].set_xlabel('iteration')
        axs[1][1].set_ylabel(r'G')

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
        fig: plt.Figure = plt.figure(figsize=(9, 5))
        axs: list[plt.Axes] = fig.subplots(1, 3)
        fig.subplots_adjust(left=0.05, right=0.97, top=1, bottom=0.15)
        Ni_pch = self.N_f[range(idx, self.n * self.S, self.S)]
        Gi_pch = self.G[range(idx, self.n * self.S, self.S)]

        def plotHeatmap(ax: plt.Axes, to_plot, title, xlabel='Arrival Patch', ylabel='Departure Patch'):
            im = ax.imshow(to_plot)
            cbar = ax.figure.colorbar(im, ax=ax, anchor=(0, 0.5), shrink=0.5)
            for k in range(self.n):
                for l in range(self.n):
                    text = ax.text(l, k, round(to_plot[k, l], 4),
                                   ha="center", va="center", color="w")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            secax_x1 = ax.secondary_xaxis(-0.2)
            secax_x1.set_xlabel(r'$N_i$ in each patch')
            secax_x1.set_xticks(range(self.n), np.round(Ni_pch, 4))

            secax_x2 = ax.secondary_xaxis(-0.4)
            secax_x2.set_xlabel(r'$G_i$ in each patch')
            secax_x2.set_xticks(range(self.n), np.round(Gi_pch, 4))

        plotHeatmap(axs[0], self.alpha_species[idx], title=r'$\alpha^{ij}$ of species ' + f'#{idx}')
        plotHeatmap(axs[1], self.singleflow[idx], title=f'Single-flow of species #{idx}')
        plotHeatmap(axs[2], self.flow[idx], title=f'Net-flow of species #{idx}')

    def ploteigval(self):
        """
        plot the eigenvalues of Jacobian matrix on the complex plane
        :return:
        """
        x = self.eigval.real
        x_max = max(x)
        y = self.eigval.imag

        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        ax.scatter(x, y)
        ax.set_title('Eigenvalues of Jacobian matrix')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        # ax.set_xlim(-20, 5)
        # ax.set_ylim(-10, 10)
        ax.axvline(x=x_max, ls='--', c='r')
        # ax.set_aspect('equal')

    def ploteigval2(self):
        fig: plt.Figure = plt.figure(figsize=(15, 6))
        axs: list[plt.Axes] = fig.subplots(1, 2,
                                           gridspec_kw={'left': 0.07, 'right': 0.96})
        fig.subplots_adjust(wspace=0.2, hspace=0.35)
        fig.suptitle(r'Real Part of the Leading Eigenvalue')

        axs[0].scatter(range(len(self.maxFactEigvals)), self.maxFactEigvals)
        axs[0].set_title(r'Differential Equation')
        axs[0].set_xlabel('')
        axs[0].set_ylabel('')

        axs[1].scatter(range(len(self.maxTheoryEigvals)), self.maxTheoryEigvals)
        axs[1].set_title(r'Algebraic Equation')
        axs[1].set_xlabel('')
        axs[1].set_ylabel('')

    def plotErrors(self):
        fig: plt.Figure = plt.figure(figsize=(15, 6))
        ax = fig.subplots(1, 1)
        ax.plot(range(len(self.errors)), self.errors)
        ax.set_title('Errors between Theory and Fact')

    def plotAll(self, histN=False, histAfa=False, flow=False):
        """
        plot: N, eigenvalues, parameters
        :return:
        """
        plt.rcParams.update({'font.size': 12})
        self.plotParam()
        self.plotNandAlpha()
        self.ploteigval()
        self.ploteigval2()
        self.plotErrors()
        if histN:
            self.plotHist_N()
        if histAfa:
            self.plotHist_Afa()
        if flow:
            for i in range(self.S):
                self.plotFlow(idx=i)
                plt.show()

        plt.show()
