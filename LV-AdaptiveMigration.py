import copy
import multiprocessing as mp
import os
import pickle
import time
from tkinter import filedialog

import numba
import numpy as np
import scipy as sp
import sympy as smp
from enum import Enum
from matplotlib import pyplot as plt


class NetParam(Enum):
    """
    Define the name of the network parameters.
    """
    species = {'latex': r'$S$', 'str': 'S'}  # number of species
    patches = {'latex': r'$n$', 'str': 'n'}  # number of patches
    randomSeed = {'latex': 'randomSeed', 'str': 'seed'}  # seed of random numbers

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

    method_alpha: dict = {'name': r'model of $\alpha$', 0: 'Personal', 1: 'Softmax'}

    rho = {'latex': r'$\rho$', 'str': 'rho'}  # correlation between patches
    n_e = {'latex': r'$n_{e}$'}  # the effective number of ecologically independent patches in the meta-ecosystem
    left1 = {'latex': r'$\sigma \sqrt{c \left (S-1 \right ) } $'}  # the left term of May's inequality
    left2 = {'latex': r'$\sigma \sqrt{c \left (S-1 \right ) /n_{e}  } $'}  # the left term of May's inequality

    method_ode = 'computation method'  # numerical computation method
    dt = {'latex': 'dt'}  # time interval
    maxIteration = {'latex': 'maxIteration'}  # max iteration time
    maxError = {'latex': 'maxError'}  # max error in iteration


class AdaptiveMigration:
    def __init__(self, config: dict):
        self.seed = config[NetParam.randomSeed]
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

        self.d = config[NetParam.dispersal]
        self.kappa = config[NetParam.kappa]

        self.method_alpha = config[NetParam.method_alpha]
        self.method_ode = config[NetParam.method_ode]
        self.dt = config[NetParam.dt]  # time interval
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

        # print(self.A, self.M)

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
            # self.rho = 1 / np.sqrt(1 + (self.sgm_qx / self.sgm_aij) ** 2)
            self.sgm_qx = self.sgm_aij * np.sqrt(1 / self.rho - 1)
            lmd = 1 / np.sqrt(1 + (self.sgm_qx / self.sgm_aij) ** 2)
            self.n_e = self.n / (1 + (self.n - 1) * self.rho)
            self.left1 = self.sgm_aij * np.sqrt(self.c * (self.S - 1))
            self.left2 = self.sgm_aij * np.sqrt(self.c * (self.S - 1) / self.n_e)
            # print(f'left1: {self.left1}')

            A_std = np.random.normal(0, self.sgm_aij, size=(self.S, self.S))
            Connect = np.random.binomial(1, self.c, size=(self.S, self.S))
            for i in range(self.n):
                same = np.random.normal(0, self.sgm_qx, size=(self.S, self.S))
                Ax = (A_std + same) * lmd
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
                # # im, cbar = heat
                # im = ax.imshow(Ax)
                # for i in range(self.S):
                #     for j in range(self.S):
                #         text = ax.text(j, i, round(Ax[i, j], 2),
                #                        ha="center", va="center", color="w")
                # fig.tight_layout()
                # plt.show()
                # plt.imshow(A_std)
                # plt.hist(A_std)

                self.Ax_lst.append(Ax)
                # lst_A.append(A_std)
            # print(lst_A[1] - lst_A[0])

            self.A = sp.linalg.block_diag(*self.Ax_lst)

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

    def ode(self, t=0, N=0, Alpha=0, mode=2):
        """
        the preference-dispersal LV equation represented with matrix
        :param N: current number matrix of species
        :return: dN / dt
        """
        # Compute N
        self.G = self.M + np.matmul(self.A, N)
        term_N_1 = np.multiply(N, self.G)

        N_patch_lst = [N[i: i + self.S] for i in range(0, self.n * self.S, self.S)]
        Ndiag = [np.diag(np.concatenate([v] * self.n)) for v in N_patch_lst]
        P1 = np.concatenate(Ndiag, axis=1)

        term_N_2 = np.matmul(P1, Alpha)
        term_N_3 = np.multiply(N, np.matmul(self.P2, Alpha))
        result1 = term_N_1 + np.multiply(self.D, (term_N_2 - term_N_3))

        # Compute Alpha
        term_Afa_1 = np.multiply(self.K, Alpha)
        P3 = np.concatenate([self.G] * self.n)

        G_patch_lst = [self.G[i: i + self.S] for i in range(0, self.n * self.S, self.S)]
        Gdiag_row = np.concatenate([np.diag(v) for v in G_patch_lst], axis=1)
        Gdiag = np.concatenate([Gdiag_row] * self.n, axis=0)
        P4 = sp.linalg.block_diag(*[Gdiag] * self.n)

        if mode == 1:
            self.avgG = np.matmul(P4, Alpha)
            term_Afa_2 = P3 - self.avgG
        elif mode == 2:
            self.avgG = np.matmul(P4, np.ones(shape=self.S * self.n * self.n)) / self.n
            term_Afa_2 = P3 - self.avgG

        result2 = np.multiply(term_Afa_1, term_Afa_2)

        return result1, result2

    # @numba.jit()
    def odeMougi(self, N=0):
        """
        the preference-dispersal LV equation represented with matrix
        :param N: current number matrix of species
        :return: dN / dt
        """
        self.curAlpha = self.calcAlpha(N)
        term_1 = N * self.G

        N_patch_lst = [N[i: i + self.S] for i in range(0, self.n * self.S, self.S)]
        Ndiag = [np.diag(np.concatenate([v] * self.n)) for v in N_patch_lst]
        P1 = np.concatenate(Ndiag, axis=1)

        term_N_2 = np.matmul(P1, self.curAlpha)
        term_N_3 = np.multiply(N, np.matmul(self.P2, self.curAlpha))
        term_flow = np.multiply(self.D, (term_N_2 - term_N_3))
        result1 = term_1 + term_flow

        # pool = multiprocessing.Pool(2).map(self.odeMougiMultiprocessing, [(1, N), (2, N)])
        # result1 = sum(pool)

        # if self.iter > 45000 and self.iter % 1000 == 0:
        #     print('G:', self.G)
        #     print('term_N_1:', term_N_1)
        #     print('term_flow:', term_flow)
        #     print('dN/dt:', result1)

        return result1

    def calcAlpha(self, N):
        self.G = self.M + np.matmul(self.A, N)
        P3 = np.concatenate([self.G] * self.n)
        G_species_lst = [self.G[i::self.S] for i in range(0, self.S)]
        temp = [np.sum(np.exp(self.kappa * Gi)) for Gi in G_species_lst]
        sumExp = np.concatenate([temp] * self.n ** 2)
        Alpha = np.exp(self.kappa * P3) / sumExp
        return Alpha

    # @numba.jit(nopython=True)
    def ode_spRK4(self, t=0, y0: np.ndarray = None):
        if self.method_ode == 1:
            N, Alpha = y0[: self.S * self.n], y0[self.S * self.n:]
            return np.concatenate(self.ode(N=N, Alpha=Alpha))
        else:
            return self.odeMougi(N=y0)

    def RK4(self, N, Alpha, dt, method_Alpha=1):
        """
        4th-order Runge-Kutta method
        :param N: current number matrix of species
        :param dt: time interval
        :return: N_n+1
        """
        if method_Alpha == NetParam.method_alpha.value[0]:
            k11, k12 = self.ode(N=N, Alpha=Alpha)
            k21, k22 = self.ode(N=N + k11 * dt / 2, Alpha=Alpha + k12 * dt / 2)
            k31, k32 = self.ode(N=N + k21 * dt / 2, Alpha=Alpha + k22 * dt / 2)
            k41, k42 = self.ode(N=N + k31 * dt, Alpha=Alpha + k32 * dt)

            result1 = N + dt * (k11 + 2 * k21 + 2 * k31 + k41) / 6
            result2 = Alpha + dt * (k12 + 2 * k22 + 2 * k32 + k42) / 6
            self.curN, self.curAlpha = result1, result2

        elif method_Alpha == NetParam.method_alpha.value[1]:
            k1 = self.odeMougi(N=N)
            k2 = self.odeMougi(N=N + k1 * dt / 2)
            k3 = self.odeMougi(N=N + k2 * dt / 2)
            k4 = self.odeMougi(N=N + k3 * dt)

            slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # if self.iter > 45000 and self.iter % 1000 == 0:
            #     print('slope:', (k1 + 2 * k2 + 2 * k3 + k4) / 6)
            #
            # if self.iter >= 4000:
            #     self.sumslope += slope
            #
            # if self.iter >= 1e4:
            #     self.slopes.append(k1[:5].reshape(1, -1))

            self.curN = N + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def odeMougiSymbol(self, t=0, N=0):
        """
        the preference-dispersal LV equation represented with matrix
        :param N: current number matrix of species
        :return: dN / dt
        """
        # Compute N
        self.G = self.M + np.matmul(self.A, N)
        term_N_1 = np.multiply(N, self.G)

        P3 = np.concatenate([self.G] * self.n)
        expG_species_lst = [list(map(smp.exp, self.kappa * self.G[i::self.S])) for i in range(0, self.S)]
        temp = [sum(expGi) for expGi in expG_species_lst]
        sumExp = np.concatenate([temp] * self.n ** 2)

        self.curAlpha = np.array(list(map(smp.exp, self.kappa * P3))) / sumExp
        self.curAlpha = smp.simplify(self.curAlpha)

        N_patch_lst = [N[i: i + self.S] for i in range(0, self.n * self.S, self.S)]
        Ndiag = [np.diag(np.concatenate([v] * self.n)) for v in N_patch_lst]
        P1 = np.concatenate(Ndiag, axis=1)

        term_N_2 = np.matmul(P1, self.curAlpha)
        term_N_3 = np.multiply(N, np.matmul(self.P2, self.curAlpha))
        term_flow = np.multiply(self.D, (term_N_2 - term_N_3))
        result1 = term_N_1 + term_flow

        # if self.iter > 45000 and self.iter % 1000 == 0:
        #     print('G:', self.G)
        #     print('term_N_1:', term_N_1)
        #     print('term_flow:', term_flow)
        #     print('dN/dt:', result1)

        return result1

    def searchTheoryPoint(self, N_guess):
        print('start search')
        root = sp.optimize.fsolve(self.odeMougi, N_guess, maxfev=int(1e6))
        print('search end')
        return root

    def theoryPoint(self, x0, procs):
        pool = mp.Pool(min(10, procs))
        results = pool.map(self.searchTheoryPoint, x0)
        realx0 = []
        for id, root in enumerate(results):
            if np.linalg.norm(self.odeMougi(root)) > 1e-4:
                continue
            if sum(root <= 1e-4) > 0:
                continue
            # if sum(root < 0) > 0:
            #     continue
            for item in self.theoryPoints:
                if np.linalg.norm(root - item) <= 1e-4:
                    break
                else:
                    print(sum(abs(root - item)) / (self.S * self.n))
                    print(np.linalg.norm(root - item))
            else:
                realx0.append(x0[id])
                self.theoryPoints.append(root)
            # self.theoryPoints.append(root)

        return realx0

        # for N_guess in x0:
        #     root = sp.optimize.fsolve(self.odeMougi, N_guess)
        #     if sum(root <= 0) > 0:
        #         continue
        #     for item in self.theoryPoints:
        #         if np.all(root == item):
        #             break
        #     else:
        #         self.theoryPoints.append(root)

        # N = np.array(smp.symbols(f'N0:{self.S*self.n}'))
        # print('start solve')
        # solution = smp.solve(self.odeMougiSymbol(N=N))
        # print('end solve')
        # for solu in solution:
        #     point = np.array(list(solu.values()), dtype=complex)
        #     if sum(point <= 0) == 0:
        #         self.theoryPoints.append(point)

    # @numba.jit()
    def findFixed(self, method_ode=1):
        """
        Find the fixed point.Record species numbers N in every 2 steps.
        Stop computing if iteration exceeds maxIteration.
        :return:
        """
        self.pst, self.stable = True, False
        self.iter_lst, self.N_lst, self.Alpha_lst, self.G_lst, self.avgG_lst = [], [], [], [], []

        if method_ode == 1:
            self.curN, self.curAlpha = self.N0, self.Alpha0
            index = range(0, self.S * self.n, self.S)
            iteration = 0
            while True:  # sum(abs(N_new - N_old) > self.maxError) > 0:
                # old_N, old_Alpha = self.curN, self.curAlpha
                self.RK4(self.curN, self.curAlpha, self.dt, self.method_alpha)
                self.iter = iteration
                iteration += 1

                if iteration % 2 == 0:
                    self.iter_lst.append(iteration)
                    self.N_lst.append(self.curN.reshape(1, -1))
                    self.Alpha_lst.append(self.curAlpha.reshape(1, -1))

                    self.G_lst.append(self.G.reshape(1, -1))
                    self.avgG_lst.append(self.avgG.reshape(1, -1))

                if iteration % 1000 == 0:
                    print(iteration)

                # if some Ni <= 0, the fixed point does not exist
                if sum(self.curN <= 1e-13) > 0 | sum(self.curAlpha < 0) > 0:
                    self.pst = False
                    break
                # Stop if iteration time exceeds the maximum
                if iteration > self.maxIter:
                    # N_new = None
                    # self.unstableReason = 'iteration overflow'
                    break

            self.N_f, self.Alpha_f = self.curN, self.curAlpha

            # plt.plot(range(10000, int(self.maxIter)), np.concatenate(self.slopes, axis=0))
            # plt.show()

        elif method_ode == 2:
            self.curN, self.curAlpha = self.N0, self.Alpha0
            if self.method_alpha == NetParam.method_alpha.value[0]:
                param = [self.ode_spRK4, np.concatenate([self.N0, self.Alpha0])]
            elif self.method_alpha == NetParam.method_alpha.value[1]:
                param = [self.ode_spRK4, self.N0]

            solver = sp.integrate.RK45(param[0], t0=0, y0=param[1], t_bound=self.dt * self.maxIter,
                                       max_step=100 * self.dt)

            last_N = self.N0
            for i in range(int(self.maxIter)):
                if i % 2 == 0:
                    self.iter_lst.append(solver.t / self.dt)
                    if self.method_alpha == NetParam.method_alpha.value[0]:
                        self.curN = solver.y[: self.S * self.n]
                        self.curAlpha = solver.y[: self.S * self.n]
                    elif self.method_alpha == NetParam.method_alpha.value[1]:
                        self.curN = solver.y

                    self.N_lst.append(self.curN.reshape(1, -1))
                    self.Alpha_lst.append(self.curAlpha.reshape(1, -1))
                    self.G_lst.append(self.G.reshape(1, -1))
                    self.avgG_lst.append(self.avgG.reshape(1, -1))

                solver.step()

                if i % 1000 == 0:
                    print(i)

                if (sum(self.curN <= 1e-4) > 0) | (sum(self.curAlpha < 0) > 0):
                    # print(i)
                    self.pst = False
                    break

                if (i % 100 == 0) and (i != 0):
                    rtol = abs(self.curN - last_N) / last_N
                    # if all the relevant tolerance < 1e-4, regard the system as stable
                    if sum(rtol >= 1e-4) == 0:
                        self.stable = True
                        break
                    # if not stable, update last_N
                    last_N = self.curN

                if solver.status == 'finished':
                    self.stable = True
                    break

                if solver.status == 'failed':
                    self.stable, self.pst = False, False
                    return

            self.N_f, self.Alpha_f = self.curN, self.curAlpha
            # print('minN:', np.min(self.N_f))
            print('persistence:', self.pst)
            print('stable:', self.stable)

    def analysis(self):
        """
        compute the flow of each species
        :return:
        """
        for i in range(self.S):
            Ni_spc = self.N_f[range(i, self.n * self.S, self.S)]
            Ni_temp = np.concatenate([Ni_spc.reshape(-1, 1)] * self.n, axis=1)
            Alpha_spc = self.Alpha_f[range(i, self.n * self.n * self.S, self.S)].reshape(self.n, self.n)
            single_flow = np.multiply(Ni_temp, Alpha_spc)
            net_flow = self.d * (single_flow - single_flow.T)

            self.var_Nf.append(np.var(Ni_spc))

            self.alpha_species.append(Alpha_spc)
            self.var_Alphaf.append(np.var(Alpha_spc))

            self.singleflow.append(single_flow)
            self.flow.append(net_flow)
            self.var_flow.append(np.var(net_flow))

            self.absflow.append(np.sum(np.abs(net_flow.flatten())))

            entropy_spc = sum(sum(np.multiply(-Alpha_spc, np.log(Alpha_spc)))) / self.n
            self.entropy.append(entropy_spc)

        self.var_Nf.append(np.var(self.N_f))
        self.var_Alphaf.append(np.var(self.Alpha_f))
        self.absflow.append(np.average(self.absflow))

        self.N_var = [np.var(self.N_f[i:i + self.S]) for i in range(0, self.n * self.S, self.S)]

    def Jacobian(self, Nf, Afaf, theory=False):
        S, n = self.S, self.n
        j1_1 = self.kappa
        j1_2 = np.concatenate(self.Ax_lst, axis=1)
        j1_2 = np.concatenate([j1_2] * n, axis=0)
        j1_3 = np.concatenate([Afaf[:S * n].reshape(-1, 1)] * S * n, axis=1)
        temp = [Afaf[i: i + S].reshape(-1, 1) for i in range(0, S * n, S)]
        temp2 = [np.concatenate([item] * S, axis=1) for item in temp]
        j1_4_1 = np.concatenate([np.concatenate(temp2, axis=1)] * n, axis=0)
        one = [np.ones(shape=(S, S))] * n
        j1_4_2 = sp.linalg.block_diag(*one)
        j1_4 = j1_4_2 - j1_4_1
        Nspc = [Nf[i: S * n: S] for i in range(S)]
        Nsum = np.array(list(map(sum, Nspc))).reshape(-1, 1)
        j1_5 = np.concatenate([np.concatenate([Nsum] * n * S, axis=1)] * n, axis=0)
        J1 = j1_1 * j1_2 * j1_3 * j1_4 * j1_5

        j2_1 = [np.diag(Afaf[i: i + S]) for i in range(0, S * n * n, S)]
        j2_2 = [np.concatenate(j2_1[i: i + n], axis=0) for i in range(0, n * n, n)]
        j2_3 = np.concatenate(j2_2, axis=1)
        J2 = j2_3 - np.identity(S * n)

        N_diag = np.diag(Nf.reshape(self.n * self.S))
        temp1 = np.matmul(N_diag, self.A)
        temp2 = np.diag(self.M + np.matmul(self.A, Nf))
        J3 = temp1 + temp2

        if theory:
            J = J3 + self.d * (J1 + J2)
            eigval = np.linalg.eigvals(J)
            self.maxTheoryEigvals.append(max(eigval.real))
        else:
            self.J = J3 + self.d * (J1 + J2)
            self.eigval = np.linalg.eigvals(self.J)
            self.maxEigval = max(self.eigval.real)
            self.maxFactEigvals.append(max(self.eigval.real))
            self.fixpt.append(Nf)

    def compute(self):
        """
        compute the fixed point
        :return:
        """
        self.spawn()

        self.findFixed(method_ode=self.method_ode)
        self.Jacobian(self.N_f, self.Alpha_f)

        procs = 10
        x0 = [np.random.random(size=self.S * self.n) * 10 for i in range(procs)]
        # x0 = [np.random.normal(1, 0.3, size=self.S * self.n) for i in range(procs)]
        # x0.append(self.N_f + np.random.normal(0, 0.1, size=self.S * self.n))
        # x0 = []
        x0.append(self.N0)
        # print(x0)

        # x0 = self.theoryPoint(x0, procs)
        # print('theorypoints:', self.theoryPoints)
        # for theoryPoint in self.theoryPoints:
        #     self.Jacobian(theoryPoint, self.calcAlpha(theoryPoint), theory=True)

        # for id, iv in enumerate(x0):
        #     self.N0 = iv
        #     # print(self.N0)
        #     self.findFixed(method_ode=self.method_ode)
        #     self.errors.append(sum(abs(self.N_f - self.theoryPoints[id])))
        #     if self.pst and self.stable:
        #         self.Jacobian(self.N_f, self.Alpha_f)
        # print('errors')
        # for item in self.fixpt:
        #     print(sum(abs(item - self.fixpt[0])))

        # for tp in self.theoryPoints:
        #     # self.N0 = tp + np.random.normal(0, 0.1, size=self.S * self.n)
        #     self.N0 = tp
        #     # print(self.N0)
        #     self.findFixed(method_ode=self.method_ode)
        #     if self.pst and self.stable:
        #         self.Jacobian(self.N_f, self.Alpha_f)
        #     # self.plotNandAlpha()
        #     # plt.show()

        # self.theoryPoint(x0, procs)
        # # print(self.theoryPoints)
        # for theoryPoint in self.theoryPoints:
        #     self.Jacobian(theoryPoint, self.calcAlpha(theoryPoint), theory=True)

        # print(self.odeMougi(self.N_f))
        self.analysis()
        # self.Jacobian(self.N_f, self.Alpha_f)

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
        axs: plt.Axes = fig.subplots(1, 3)
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


class NetworkSweeper:
    """
    This class works as an interface to compute networks.
    """

    def __init__(self, species=26, patches=5, randomSeed=0,
                 initial='random', growth=1, n0=1, alpha0=0.2, N0=None, Alpha0=None,
                 miu_n0=1, sgm_n0=0.05, miu_alpha=0.5, sgm_alpha=0.01,
                 connectance=0.6, sgm_aij=0.1, sgm_qx=10, rho=0.1, Adiag=1, dispersal=0, kappa=1,
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

        self.ini_dpd = ini_dpd
        self.runtime = runtime

        self.unchange_var = None
        self.change_var = {'var': NetParam.randomSeed, 'value': []}

        self.var_Nf, self.var_Alphaf, self.var_flow, self.absflow = [], [], [], []
        self.maxmin_Nf, self.maxmin_Alphaf, self.maxEigvals = [], [], []
        self.entropy = []
        self.Nf_lst = []

        self.pst = []

        self.aim = ''

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
        net = AdaptiveMigration(self.config)
        net.compute()
        # if not net.stable:
        #     continue
        # self.net_lst.append(net)
        net.disp()
        net.plotAll()

    def plotaxes(self, X, Y, ax: plt.Axes, twinx=False, **kwargs):
        if twinx:
            ax = ax.twinx()
        lines = ax.plot(X, Y)

        if 'dashes' in kwargs:
            for line in lines:
                line.set_dashes(kwargs['dashes'])
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
        if 'xilm' in kwargs:
            ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

        return lines

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
            net = AdaptiveMigration(self.config)
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
            net = AdaptiveMigration(self.config)
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
        title = r'Dependency of $N_{f}$ and $\alpha_{f}$ on initial value' + f'\nGeneration method: {models[mode]}'
        # title = r'First group of $\alpha:$'\
        #         + r'$\left ( \alpha _{1}^{11} ,\alpha _{1}^{12},\dots \alpha _{1}^{1n} \right )$'\
        #         + f'generation method: {models[mode]}'

        fig.suptitle(title)

        index = range(0, S * n, S)
        print(Alpha0_lst[0][index])
        # _ = axs[0].hist(Alpha0_lst[0][index], range=(0, 1), bins=30, weights=None)
        # plt.show()
        # temp = [Alpha0_lst[i][index].reshape(1, -1) for i in range(runtime)]
        # self.plotaxes(X=xAfa, Y=np.concatenate(temp, axis=0), ax=axs[0],
        #               xlabel='runs', ylabel=r'$\alpha_{0}$', ylim=(0, 1))

        net = None
        N1_lst, Alpha1_lst = [], []
        for Alpha0 in Alpha0_lst:
            self.config[NetParam.N0] = None
            self.config[NetParam.Alpha0] = Alpha0
            # _ = axs[0].hist(Alpha0_lst[0], range=(0, 1), bins=30, weights=None)
            net = AdaptiveMigration(self.config)
            net.compute()
            # net.plotNandAlpha()
            # plt.show()
            N1_lst.append(net.N_f.reshape(1, -1))
            # Alpha1_lst.append(net.Alpha_f[index].reshape(1, -1))
            Alpha1_lst.append(net.Alpha_f.reshape(1, -1))

        Y1, Y2 = np.concatenate(N1_lst, axis=0), np.concatenate(Alpha1_lst, axis=0)
        self.plotaxes(X=xAfa, Y=Y1, ax=axs[0], xlabel='runs', ylabel=r'$N_{f}$',
                      ylim=(np.min(Y1) * 0.9, np.max(Y1) * 1.1))
        self.plotaxes(X=xAfa, Y=Y2, ax=axs[1], xlabel='runs', ylabel=r'$\alpha_{f}$',
                      ylim=(np.min(Y2) * 0.9, np.max(Y2) * 1.1))

        # net.plotAll(histN=False, histAfa=False, flow=False)
        net.plotParam()

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

    def sweep(self, unchange: NetParam, value_unchg, change: NetParam, values_chg: list,
              research_id: int = 0, envs=None, seed=1):
        self.__init__()
        self.aim = 'sweep'
        self.config[NetParam.randomSeed] = seed
        self.config[unchange] = value_unchg
        self.unchange_var = unchange
        self.change_var['var'] = change
        self.change_var['value'] = values_chg
        # research = ['LinearStability vs d, kpa']
        # env = ''
        # for item in envs:
        #     env += f'{item.value["str"]}{self.config[item]},'
        #
        # if self.ini_dpd:
        #     directory = './log/N0 initial dependency'
        #     S, n = self.config[NetParam.species], self.config[NetParam.patches]
        #     np.random.seed(1)
        #     self.change_var['value'] = [np.random.normal(1, 0.3, size=S * n) for i in range(self.runtime)]
        # else:
        #     directory = './log/sweep/' \
        #                 + f'{research[research_id]}/' \
        #                 + f'{env}/' \
        #                 + f'{unchange.value["str"]}{value_unchg},' \
        #                 + f'{change.value["str"]}{str(min(values_chg))}~{str(max(values_chg))}'
        #
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        converge = True
        err_lst = []
        net_lst = []
        for value in self.change_var['value']:
            self.config[change] = value
            net = AdaptiveMigration(self.config)
            net.compute()
            if not net.stable or not net.pst:
                converge = False
                err_lst.append({
                    'seed': self.config[NetParam.randomSeed],
                    'S': self.config[NetParam.species],
                    'n': self.config[NetParam.patches],
                    'd': self.config[NetParam.dispersal],
                    'kpa': self.config[NetParam.kappa],
                    'c': self.config[NetParam.connectance],
                    'rho': self.config[NetParam.rho],
                    'sgm': self.config[NetParam.sgm_aij]
                })
            net_lst.append(net)
            self.var_Nf.append(net.var_Nf)
            self.maxmin_Nf.append([np.max(net.N_f), np.min(net.N_f)])
            self.var_Alphaf.append(net.var_Alphaf)
            self.maxmin_Alphaf.append([np.max(net.Alpha_f), np.min(net.Alpha_f)])
            self.var_flow.append(net.var_flow)
            self.entropy.append(net.entropy)
            self.absflow.append(net.absflow)
            self.Nf_lst.append(net.N_f)
            self.maxEigvals.append(net.maxEigval)

        return converge, err_lst

        # with open(directory + '/runlog.txt', 'w') as f:
        #     msg = f'all converged: {converge}\nerrors: {err_lst}'
        #     f.write(msg)
        #
        # filename1 = directory + '/AllNets.bin'
        # DataLoader.store(filename1, *net_lst)
        #
        # filename2 = directory + '/SweeperResult.bin'
        # DataLoader.store(filename2, self)
        #
        # print('sweep finished')

    def linearStability(self, unchange: NetParam, unchg_value: list, change: NetParam, chg_value: list, seeds=50):
        vars = [NetParam.species, NetParam.patches, NetParam.connectance, NetParam.rho, NetParam.sgm_aij,
                NetParam.dispersal, NetParam.kappa]
        vars.remove(unchange)
        vars.remove(change)
        directory = "./log/sweep/LinearStability/"
        directory += f"{unchange.value['str']}{min(unchg_value)}~{max(unchg_value)}"
        directory += f"{change.value['str']}{min(chg_value)}~{max(chg_value)}"
        for var in vars:
            directory += f"{var.value['str']}{self.config[var]}"
        # directory = "./log/sweep/LinearStability vs d, kpa/" \
        #             + f"c{self.config[NetParam.connectance]}rho{self.config[NetParam.rho]}" \
        #             + f"sgm{self.config[NetParam.sgm_aij]}" \
        #             + f"S{self.config[NetParam.species]}n{self.config[NetParam.patches]}" \
        #             + f"d{min(d_value)}~{max(d_value)}" \
        #             + f"kpa{min(kpa_value)}~{max(kpa_value)}seed0~{seeds}"
        to_file = []
        flag_converge = True
        errors = []
        for v in unchg_value:
            sweeper_lst = []
            print(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)

            for seed in range(seeds):
                converge, err_lst \
                    = self.sweep(unchange=unchange, value_unchg=v, change=change, values_chg=chg_value, seed=seed)
                if not converge:
                    flag_converge = False
                    errors.extend(err_lst)
                sweeper_lst.append(copy.copy(self))
            to_file.append(sweeper_lst)

        with open(f'{directory}/runlog.txt', 'w') as f:
            msg = f'all converged: {flag_converge}\nerrors: {errors}'
            f.write(msg)

        DataLoader.store(f'{directory}/result.bin', *to_file)

        print('sweep finished')

    def persistence(self, runs, change: NetParam, values_chg: list):
        self.aim = 'pst'
        self.change_var['var'] = change
        self.change_var['value'] = values_chg

        directory = f'./log/persistence/pst vs. {self.change_var["var"].value["str"]}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        seeds = range(int(runs))
        fails = []
        for value in values_chg:
            print(f'{change.value["str"]}', value)
            self.config[change] = value
            pst = 0
            total = 0
            for seed in seeds:
                print('seed:', seed)
                np.random.seed(seed)
                m = np.random.uniform(0, 10)
                print(m)
                self.config[NetParam.randomSeed] = seed
                self.config[NetParam.growth] = m
                self.config[NetParam.n0] = m
                net = AdaptiveMigration(self.config)
                try:
                    net.compute()
                    total += 1
                    if net.pst:
                        pst += 1
                except:
                    fails.append({'m': m, 'param': change.value['str'], 'value': value, 'seed': seed})
                    pass

            self.pst.append(pst / total)
        print('fails:', fails)

        filename = directory + f'/{change.value["str"]}{str(min(values_chg))}~{str(max(values_chg))}'
        DataLoader.store(filename, self)

    def plotSweep(self):
        # fig = plt.Figure()
        # axs = fig.subplots(2, 3)
        fig, axs = plt.subplots(2, 3)
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.07, top=0.9, wspace=0.3)

        if self.ini_dpd:
            X = range(self.runtime)
            xlabel = 'runs'
        else:
            X = np.array(self.change_var['value'])
            xlabel = self.change_var['var'].value['latex']
        Xs = [[X] * 3, [X] * 3]
        xlabels = [[xlabel] * 3, [xlabel] * 3]
        ylabels = [[
            r'$\sigma^2(N_f)\times 10$',
            [r'$\sigma^2(N_f)\times 10$', r'$\sigma^2(\alpha_f)\times 10^3$'],
            r'$\ln{\sigma^2(f_{net})}$'
        ], [
            r'$\sigma^2(\alpha_f)\times 10^3$',
            [r'$N_f$', r'$\alpha_f$'],
            r'$S_i$'
        ]]

        self.var_Nf = np.array(self.var_Nf) * 10
        self.var_Alphaf = np.array(self.var_Alphaf) * 1e3
        self.entropy = np.array(self.entropy)

        self.plotaxes(Xs[0][0], self.var_Nf[:, :-1], axs[0][0],
                      title=r'$\sigma^2(N_f)$ vs. $\kappa$', xlabel=xlabels[0][0], ylabel=ylabels[0][0])

        lines1 = self.plotaxes(Xs[0][1], self.var_Nf[:, -1], axs[0][1], legend=['variance of all N'],
                               xlabel=xlabels[0][1], ylabel=ylabels[0][1][0], title=r'variance of N and $\alpha$')
        lines2 = self.plotaxes(Xs[0][1], self.var_Alphaf[:, -1], axs[0][1], twinx=True, ylabel=ylabels[0][1][1],
                               dashes=[2, 2, 10, 2], legend=[r'variance of all $\alpha$'])
        axs[0][1].legend(lines1 + lines2, ['variance of all N', r'variance of all $\alpha$'])

        self.plotaxes(Xs[0][2], np.log(self.var_flow), axs[0][2],
                      title=r'$\sigma^2(f_{net})$ vs. $\kappa$', xlabel=xlabels[0][2], ylabel=ylabels[0][2])

        self.plotaxes(Xs[1][0], self.var_Alphaf[:, :-1], axs[1][0],
                      title=r'$\sigma^2(\alpha_f)$ vs. $\kappa$', xlabel=xlabels[1][0], ylabel=ylabels[1][0])

        lines1 = self.plotaxes(Xs[1][1], self.maxmin_Nf, axs[1][1], title=r'Span of $N_f$ and $\alpha_f$ vs. $\kappa$',
                               xlabel=xlabels[1][1], ylabel=ylabels[1][1][0], legend=[r'$N_{max}$', r'$N_{min}$'])
        lines2 = self.plotaxes(Xs[1][1], self.maxmin_Alphaf, axs[1][1], twinx=True,
                               ylabel=ylabels[1][1][1], dashes=[2, 2, 10, 2])
        axs[1][1].legend(lines1 + lines2, [r'$N_{max}$', r'$N_{min}$'] + [r'$\alpha_{max}$', r'$\alpha_{min}$'])

        self.plotaxes(Xs[1][2], self.entropy, axs[1][2],
                      title=r'Entropy $S_i$ vs. $\kappa$', xlabel=xlabels[1][2], ylabel=ylabels[1][2])

        self.absflow = np.array(self.absflow)
        fig2, axs2 = plt.subplots(1, 3)
        self.plotaxes(X, self.absflow[:, :-1], axs2[0], xlabel=xlabel, ylabel='absflow', title='absflow per species')
        self.plotaxes(X, self.absflow[:, -1], axs2[1], xlabel=xlabel, ylabel='average flow', title='average flow')
        self.plotaxes(X, self.Nf_lst, axs2[2], xlabel=xlabel, ylabel=r'$N_f$', title=r'$N_f$ vs. runs')

        net = AdaptiveMigration(self.config)
        net.spawn()
        net.plotParam()

        fig3, axs3 = plt.subplots()
        self.plotaxes(X, self.maxEigvals, axs3, xlabel=xlabel, ylabel=r'max $Re(\lambda)$',
                      title=r'max Re($\lambda$) vs. $\kappa$')

        plt.show()

    def plotPst(self):
        fig, axs = plt.subplots()
        axs.plot(self.change_var['value'], self.pst)
        axs.set_title(f'persistence vs. {self.change_var["var"].value["latex"]}')
        axs.set_xlabel(self.change_var['var'].value["latex"])
        axs.set_ylabel('persistence')

        net = AdaptiveMigration(self.config)
        net.spawn()
        net.plotParam()

        plt.show()

    def plotAll(self):
        plt.rcParams.update({'font.size': 12})
        func = {'sweep': self.plotSweep, 'pst': self.plotPst}
        func[self.aim]()


class NetworkManager:
    def __init__(self):
        self.net_lst: list[AdaptiveMigration] = []
        self.sweeper_lst: list[NetworkSweeper] = []

    def readfile(self, filename=None):
        if filename is None:
            filename = filedialog.askopenfilename()
        data_lst = DataLoader.load(filename)
        self.net_lst = [item for item in data_lst if type(item) == AdaptiveMigration]
        self.sweeper_lst = [item for item in data_lst if type(item) == NetworkSweeper]

    def show(self, idx=0):
        if self.net_lst:
            item = self.net_lst[idx]
        else:
            item = self.sweeper_lst[0]

    def eigenvalue(self):
        dir = filedialog.askdirectory()
        y_data = []
        legends = []
        with open(f'{dir}/runlog.txt', 'r') as f:
            converge = f.readline()
            errors = f.readline()
        print(converge)
        print(errors)

        err_seeds = set()
        for err in eval(errors[7:]):
            # err_seeds.add(err[-1])
            err_seeds.add(err['seed'])

        raw_data = DataLoader.load(f'{dir}/result.bin')
        sweeper: NetworkSweeper = raw_data[0][0]
        change = sweeper.change_var['var'].value['latex']
        chg_data = sweeper.change_var['value']
        unchange = sweeper.unchange_var.value['latex']
        unchg_data = []
        data_mean = np.empty(shape=(len(raw_data), len(chg_data)))
        data_err = np.empty(shape=(len(raw_data), len(chg_data)))
        for idx, sp_lst in enumerate(raw_data):
            unchg_data.append(sp_lst[0].config[sweeper.unchange_var])
            eigvals = np.array([sweeper.maxEigvals for sweeper in sp_lst])
            eigvals = np.delete(eigvals, list(err_seeds), axis=0)
            data_mean[idx] = np.mean(eigvals, axis=0)
            data_err[idx] = np.std(eigvals, axis=0)

        lg_d = list(map(lambda x: rf'{unchange}={x}', unchg_data))
        lg_kpa = list(map(lambda x: rf'{change}={x}', chg_data))

        fig: plt.Figure = plt.figure()
        axs: list[plt.Axes] = fig.subplots(2, 1)
        ax1, ax2 = axs[0], axs[1]
        fig.suptitle(rf'max Re($\lambda$) vs. {unchange} and {change}')
        fig.subplots_adjust(hspace=0.25)
        markers = ['o', '^', 's', 'D', 'P']

        print(chg_data)
        print(data_mean)

        # for i in range(5):
        #     # ax1.plot(x_data, y1[:, i], marker=markers[i], color='k', ls='--', lw=1)
        #     ax1.plot(x_data, y1[:, i], marker=markers[i], ls='--', lw=1)
        ax1.plot(unchg_data, data_mean, 'o--')
        # for i in range(data_mean.shape[1]):
        #     ax1.errorbar(unchg_data, data_mean[:, i], data_err[:, i])
        ax1.set_xscale('symlog', linthresh=1e-4)
        ax1.legend(lg_kpa)
        ax1.set_title(rf'max $Re(\lambda)$ vs {unchange}')
        ax1.set_xlabel(rf'{unchange}')
        ax1.set_ylabel(r'max $Re(\lambda)$')

        # for i in range(data_mean.shape[0]):
        #     ax2.errorbar(chg_data, data_mean[i, :], data_err[i, :])
        ax2.plot(chg_data, data_mean.T, 'o--')
        # ax2.set_xscale('symlog')
        ax2.legend(lg_d)
        ax2.set_title(rf'max $Re(\lambda)$ vs {change}')
        ax2.set_xlabel(rf'{change}')
        ax2.set_ylabel(r'max $Re(\lambda)$')

        # for item in os.listdir(dir):
        #     check = f'{dir}/{item}/runlog.txt'
        #     with open(check, 'r') as f:
        #         if 'False' in f.readline():
        #             converge = False
        #             errors.append(item)
        #
        #     path = f'{dir}/{item}/SweeperResult.bin'
        #     self.readfile(path)
        #     eivals = [sweeper.maxEigvals for sweeper in self.sweeper_lst]
        #     y_data.append(np.mean(np.array(eivals), axis=0))
        #     legends.append(item[: item.find(',')])
        #
        # print(f'converge: {converge}')
        # print(f'errors: {errors}')

        # sweeper = self.sweeper_lst[0]
        # x_data = sweeper.change_var['value']
        # y_data = np.array(y_data).T
        # y1, y2 = y_data[:, :5], y_data[:, 4:]
        # legends.sort(key=lambda x: eval(x[2:]))
        # lg1, lg2 = legends[:5], legends[4:]
        #
        # fig: plt.Figure = plt.figure()
        # axs: list[plt.Axes] = fig.subplots(2, 1)
        # ax1, ax2 = axs
        # fig.suptitle(r'max Re($\lambda$) vs. $\kappa$ and $d$')
        # markers = ['o', '^', 's', 'D', 'P']
        #
        # for i in range(5):
        #     # ax1.plot(x_data, y1[:, i], marker=markers[i], color='k', ls='--', lw=1)
        #     ax1.plot(x_data, y1[:, i], marker=markers[i], ls='--', lw=1)
        # ax1.legend(lg1)
        # ax1.set_title(r'Small $d$')
        # ax1.set_xlabel(r'$\kappa$')
        # ax1.set_ylabel(r'max $Re(\lambda)$')
        #
        # for i in range(5):
        #     # ax1.plot(x_data, y1[:, i], marker=markers[i], color='k', ls='--', lw=1)
        #     ax2.plot(x_data, y2[:, i], marker=markers[i], ls='--', lw=1)
        # ax2.legend(lg2)
        # ax2.set_title(r'Large $d$')
        # ax2.set_xlabel(r'$\kappa$')
        # ax2.set_ylabel(r'max $Re(\lambda)$')

        net = AdaptiveMigration(sweeper.config)
        net.spawn()
        net.plotParam()
        plt.show()


class DataLoader:
    @staticmethod
    def store(file, *data):
        """
        store data to a .bin file
        :param file:
        :param data:
        :return:
        """
        with open(file, 'wb') as f:
            for item in data:
                pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file):
        """
        load data from a .bin file
        :param file:
        :return: data list
        """
        data_lst = []
        with open(file, 'rb') as f:
            while True:
                try:
                    data_lst.append(pickle.load(f))
                except EOFError:
                    break
            return data_lst


if __name__ == '__main__':
    net_sweeper = NetworkSweeper()
    net_sweeper.computeNet()
    # net_sweeper.persistence(runs=50, change=NetParam.dispersal, values_chg=list(np.arange(0.1, 2.1, 0.1)))
    # net_sweeper.change(NetParam.kappa, 0, 0.11, 0.01)
    # net_sweeper.origin(mode=1)
    # net_sweeper.origin_alpha()
    # net_sweeper = NetworkSweeper(ini_dpd=True, runtime=10)
    # N0 = [np.random.normal(1, 0.3, size=50) for i in range(10)]
    # net_sweeper.sweep(unchange=NetParam.dispersal, value_unchg=2,
    #                   change=NetParam.kappa, values_chg=[0, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 5, 7, 10])

    # d_lst = [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
    # d_lst = [0.001, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1, 3, 10]
    # for d in d_lst:
    #     # d_lst = [0.1]
    #     kpa_lst = [0, 0.1, 1, 2.5, 5, 7.5, 10, 15]
    #     net_sweeper.linearStability(unchange=NetParam.dispersal, unchg_value=[d],
    #                                 change=NetParam.kappa, chg_value=kpa_lst, seeds=1)
    # d_lst = [0.25]
    # kpa_lst = [0, 0.1, 1, 2.5, 5, 7.5, 10, 15]
    # net_sweeper.linearStability(unchange=NetParam.dispersal, unchg_value=d_lst,
    #                             change=NetParam.kappa, chg_value=kpa_lst, seeds=1)
    # rho_lst = [0.01, 0.05, 0.1, 0.3, 0.5]
    # rho_lst = [0.5, 0.7, 0.9, 1]
    # sgm_lst = [1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    # sgm_lst = [1e-3, 0.01, 0.05, 0.1, 0.2, 0.3]
    # net_sweeper.linearStability(unchange=NetParam.rho, unchg_value=rho_lst,
    #                             change=NetParam.sgm_aij, chg_value=sgm_lst, seeds=50)
    # c_lst = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # c_lst = list(np.round(np.arange(0.05, 1, 0.05), 2))
    # net_sweeper.linearStability(unchange=NetParam.rho, unchg_value=rho_lst,
    #                             change=NetParam.connectance, chg_value=c_lst, seeds=50)
    S_lst = [2, 5, 10, 20, 30, 40, 50]
    n_lst = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    S_lst = [2, 5, 8, 10, 12, 15]
    n_lst = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # net_sweeper.linearStability(unchange=NetParam.species, unchg_value=S_lst,
    #                             change=NetParam.patches, chg_value=n_lst, seeds=50)
    # net_manager = NetworkManager()
    # net_manager.readfile()
    # net_manager.show(1)

    net_manager = NetworkManager()
    net_manager.eigenvalue()
