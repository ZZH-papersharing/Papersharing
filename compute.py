import numpy as np

from network import *
import multiprocessing as mp


class Computer:
    def __init__(self, net: RandomNetwork):
        self.net = net
        self.slope = []

    def calcAlpha(self, N):
        self.net.G = self.net.M + np.matmul(self.net.A, N)
        # minute the max to avoid overflow
        maxG = self.net.G.reshape((self.net.n, self.net.S)).max(axis=0)
        G = self.net.G - np.concatenate([maxG] * self.net.n)
        # G = self.net.G
        P3 = np.concatenate([G] * self.net.n)
        G_species_lst = [G[i::self.net.S] for i in range(0, self.net.S)]
        temp = [np.sum(np.exp(self.net.kappa * Gi)) for Gi in G_species_lst]
        sumExp = np.concatenate([temp] * self.net.n ** 2)
        Alpha = np.exp(self.net.kappa * P3) / sumExp

        return Alpha

    def odeMougi(self, t=0, N=0):

        """
        the preference-dispersal LV equation represented with matrix
        :param N: current number matrix of species
        :return: dN / dt
        """
        self.net.curAlpha = self.calcAlpha(N)
        term_1 = N * self.net.G

        N_patch_lst = [N[i: i + self.net.S] for i in range(0, self.net.n * self.net.S, self.net.S)]
        Ndiag = [np.diag(np.concatenate([v] * self.net.n)) for v in N_patch_lst]
        P1 = np.concatenate(Ndiag, axis=1)

        term_N_2 = np.matmul(P1, self.net.curAlpha)
        term_N_3 = np.multiply(N, np.matmul(self.net.P2, self.net.curAlpha))
        term_flow = np.multiply(self.net.D, (term_N_2 - term_N_3))
        result1 = term_1 + term_flow
        # result1 = term_1 + self.net.D

        # self.slope.append(result1)
        self.net.G = term_1

        # result1 = np.zeros(4)
        # result1[0] = N[0] * (self.net.m + self.net.A[0][0] * N[0] + self.net.A[0][1] * N[1] + self.net.A[0][2] * N[2] + self.net.A[0][3] * N[3])
        # result1[1] = N[1] * (self.net.m + self.net.A[1][0] * N[0] + self.net.A[1][1] * N[1] + self.net.A[1][2] * N[2] + self.net.A[1][3] * N[3])
        # result1[2] = N[2] * (self.net.m + self.net.A[2][0] * N[0] + self.net.A[2][1] * N[1] + self.net.A[2][2] * N[2] + self.net.A[2][3] * N[3])
        # result1[3] = N[3] * (self.net.m + self.net.A[3][0] * N[0] + self.net.A[3][1] * N[1] + self.net.A[3][2] * N[2] + self.net.A[3][3] * N[3])

        return result1

    def findFixed(self):
        """
        Find the fixed point.Record species numbers N in every 2 steps.
        Stop computing if iteration exceeds maxIteration.
        :return:
        """
        self.net.initcompute()

        solver = sp.integrate.RK45(self.odeMougi, t0=0, y0=self.net.N0, t_bound=self.net.dt * self.net.maxIter,
                                   max_step=self.net.dt)

        last_N = self.net.N0
        slope_100step = [np.zeros(shape=self.net.S * self.net.n) for i in range(100)]
        for i in range(int(self.net.maxIter)):
            if self.net.method_alpha == NetParam.method_alpha.value[0]:
                self.net.curN = solver.y[: self.net.S * self.net.n]
                self.net.curAlpha = solver.y[: self.net.S * self.net.n]
            elif self.net.method_alpha == NetParam.method_alpha.value[1]:
                self.net.curN = solver.y

            if i % 2 == 0:
                if self.net.record:
                    # self.net.iter_lst.append(solver.t / self.net.dt)
                    self.net.iter_lst.append(i)
                    self.net.N_lst.append(self.net.curN.reshape(1, -1))
                    self.net.Alpha_lst.append(self.net.curAlpha.reshape(1, -1))
                    self.net.G_lst.append(self.net.G.reshape(1, -1))
                    self.net.avgG_lst.append(self.net.avgG.reshape(1, -1))
                    self.net.Np_lst.append(self.net.curN[0::self.net.S].reshape(1, -1))

            solver.step()

            # if i % 1000 == 0:
            #     print(i)

            if (sum(self.net.curN <= 1e-4) > 0) | (sum(self.net.curAlpha < 0) > 0):
                # print(i)
                # self.net.curN[self.net.curN <= 1e-4] = 0
                self.net.pst = False
                # break

            slope_100step[i % 100] = abs(self.odeMougi(N=self.net.curN))
            if (i % 100 == 0) and (i != 0):
                # relevant tolerance, 1e-8 is to avoid ZeroDivisionError
                # rtol = abs(self.net.curN - last_N) / (last_N + 1e-8)
                # rtol = abs(self.net.curN - last_N) / last_N
                # print('rtol:', rtol)
                # print('lastN:', last_N)
                # print('curN:', self.net.curN)
                # if all the relevant tolerance < 1e-4, regard the system as stable
                # if sum(rtol >= 1e-4) == 0:
                #     self.net.stable = True
                #     break
                # if not stable, update last_N
                last_N = self.net.curN

                if np.all(sum(slope_100step) / 100 < 1e-4):
                    self.net.stable = True
                    break
                else:
                    print(sum(slope_100step) / 100)

            if solver.status == 'finished':
                self.net.stable = True
                break

            if solver.status == 'failed':
                self.net.stable, self.net.pst = False, False
                break

        self.net.N_f, self.net.Alpha_f = self.net.curN, self.net.curAlpha

        print(self.odeMougi(N=self.net.N_f) / self.net.N_f)
        print('N_f:', self.net.N_f)
        print('iter:', i)
        print('status', solver.status)
        print('persistence:', self.net.pst)
        print('stable:', self.net.stable)

        # plt.plot(range(len(self.slope)), self.slope)
        # plt.show()

    def analysis(self):
        """
        compute the flow of each species
        :return:
        """
        for i in range(self.net.S):
            Ni_spc = self.net.N_f[range(i, self.net.n * self.net.S, self.net.S)]
            Ni_temp = np.concatenate([Ni_spc.reshape(-1, 1)] * self.net.n, axis=1)
            Alpha_spc = (self.net.Alpha_f[range(i, self.net.n * self.net.n * self.net.S, self.net.S)]
                         .reshape(self.net.n, self.net.n))
            single_flow = np.multiply(Ni_temp, Alpha_spc)
            net_flow = self.net.d * (single_flow - single_flow.T)

            self.net.var_Nf.append(np.var(Ni_spc))

            self.net.alpha_species.append(Alpha_spc)
            self.net.var_Alphaf.append(np.var(Alpha_spc))

            self.net.singleflow.append(single_flow)
            self.net.flow.append(net_flow)
            self.net.var_flow.append(np.var(net_flow))

            self.net.absflow.append(np.sum(np.abs(net_flow.flatten())))

            entropy_spc = sum(sum(np.multiply(-Alpha_spc, np.log(Alpha_spc)))) / self.net.n
            self.net.entropy.append(entropy_spc)

        self.net.var_Nf.append(np.var(self.net.N_f))
        self.net.var_Alphaf.append(np.var(self.net.Alpha_f))
        self.net.absflow.append(np.average(self.net.absflow))

        self.net.N_var = [np.var(self.net.N_f[i:i + self.net.S]) for i in range(0, self.net.n * self.net.S, self.net.S)]

    def Jacobian(self, Nf, Afaf, theory=False):
        S, n = self.net.S, self.net.n
        j1_1 = self.net.kappa
        j1_2 = np.concatenate(self.net.Ax_lst, axis=1)
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

        N_diag = np.diag(Nf.reshape(self.net.n * self.net.S))
        temp1 = np.matmul(N_diag, self.net.A)
        temp2 = np.diag(self.net.M + np.matmul(self.net.A, Nf))
        J3 = temp1 + temp2

        if theory:
            J = J3 + self.net.d * (J1 + J2)
            # J = J3
            eigval = np.linalg.eigvals(J)
            self.net.maxTheoryEigvals.append(max(eigval.real))
        else:
            self.net.J = J3 + self.net.d * (J1 + J2)
            # self.net.J = J3
            self.net.eigval = np.linalg.eigvals(self.net.J)
            self.net.maxEigval = max(self.net.eigval.real)
            self.net.maxFactEigvals.append(max(self.net.eigval.real))
            self.net.fixpt.append(Nf)

    def odeMougi_fsole(self, N):
        return self.odeMougi(t=0, N=N)

    def searchTheoryPoint(self, N_guess):
        print('start search')
        # root = sp.optimize.fsolve(self.odeMougi_fsole, N_guess, maxfev=int(1e6))
        root = sp.optimize.least_squares(self.odeMougi_fsole, N_guess, bounds=(0, np.inf))
        print('search end')
        return root

    def theoryPoint(self):
        procs = 10
        x0 = [np.random.random(size=self.net.S * self.net.n) * 10 for i in range(procs)]
        # x0 = [np.random.normal(1, 0.3, size=self.S * self.n) for i in range(procs)]
        x0.append(self.net.N_f)
        # x0 = []
        x0.append(self.net.N0)
        # print(x0)

        pool = mp.Pool(min(10, procs))
        results = pool.map(self.searchTheoryPoint, x0)
        realx0 = []
        np.set_printoptions(precision=2, linewidth=1000000)
        for id, root in enumerate(results):
            root = root.x
            dNdt = self.odeMougi_fsole(root)
            dist = np.linalg.norm(dNdt)
            print('root:', root)
            print('dNdt:', dNdt)
            print('dist:', dist)
            if dist > 1e-4:
                print('far')
                continue
            if sum(root < 0) > 0:
                print('nega')
                continue

            if len(self.net.theoryPoints) == 0:
                print('abc')
                realx0.append(x0[id])
                self.net.theoryPoints.append(root)

            for item in self.net.theoryPoints:
                if np.linalg.norm(root - item) <= 1e-4:
                    break
            else:
                realx0.append(x0[id])
                self.net.theoryPoints.append(root)

        print('theoryPoints:')
        for theoryPoint in self.net.theoryPoints:
            print(theoryPoint)
            self.Jacobian(theoryPoint, self.calcAlpha(theoryPoint), theory=True)

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


    def compute(self):
        """
        compute the fixed point
        :return:
        """
        self.findFixed()
        self.theoryPoint()
        self.Jacobian(self.net.N_f, self.net.Alpha_f)
        if self.net.record:
            self.analysis()

