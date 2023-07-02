import numpy as np
import scipy as sp


class DispersalNetwork:
    def __init__(self, randomSeed, species, patches, dispersal=1, dt=1e-2, maxIteration=1e4, maxError=1e-4):
        self.seed = randomSeed
        self.S = species  # numbers of species
        self.n = patches  # numbers of patches
        self.c = 0.2  # connectence
        self.d = dispersal  # dispersal rate
        self.N0 = None  # initial N
        self.dt = dt  # time interval
        self.maxIter = maxIteration  # max iteration time
        self.maxError = maxError  # max error in iteration

        self.M, self.A, self.C = None, None, None
        self.Nf = None  # fixed point
        self.unstableReason = None  # reason of unstable

    def spawn(self):
        """
        spawn A, M, N0\n
        A: normal distribution N(0, 1)\n
        M: (1, 1, ..., 1)
        N0: (1, 1, ..., 1)
        :return:
        """

        self._spawn_A()

        # use 1-D matrix
        self.M = np.ones(self.S * self.n)
        self.N0 = 10 * np.ones(self.S * self.n)

        temp = np.concatenate([np.identity(self.S)] * self.n, axis=1)
        self.C = np.concatenate([temp]*self.n, axis=0)

    def _spawn_A(self):
        """
        spawn the matrix A
        :return:
        """

        # spawn the matrix A_standard
        A_std = -1 * np.identity(self.S)
        for i in range(self.S):
            for j in range(self.S):
                if i != j:
                    np.random.seed(self.seed + i + j)
                    if np.random.random() < self.c:
                        A_std[i, j] = np.random.normal(0, 0.25)

        lst_A = []
        for i in range(self.n):
            np.random.seed(self.seed + i)
            same = np.random.normal(1, 0.1, size=(self.S, self.S))
            lst_A.append(np.multiply(same, A_std))

        self.A = sp.linalg.block_diag(*lst_A)

    def ode(self, t=0, N=0):
        """
        the non-dispersal LV equation represented with matrix
        :param N: current number matrix of species
        :return: dN / dt
        """
        temp1 = np.multiply(N, self.M - self.d + np.matmul(self.A, N))
        temp2 = self.d / (self.n-1) * np.matmul(self.C, N)
        return temp1 + temp2

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
        N_new = self.RK4(N_old, self.dt)
        # if all elements in |N_new - N_old| are < maxError, regard the point as a fixed point
        iteration = 0
        while sum(abs(N_new - N_old) > self.maxError) > 0:
            temp = N_new
            N_new = self.RK4(N_old, self.dt)
            N_old = temp
            iteration += 1

            # if some Ni <= 0, the fixed point does not exist
            # if sum(N_new <= 0) > 0:
            #     N_new = None
            #     self.unstableReason = 'some specie(s) extinguish'
            #     break
            # if iteration time exceeds the maximum, the fixed point does not exist
            if iteration > self.maxIter:
                N_new = None
                self.unstableReason = 'iteration overflow'
                break

        self.Nf = N_new

    def compute(self):
        """
        compute the fixed point
        :return:
        """
        self.spawn()
        self.findFixed()

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
        print(f'Reason: {self.unstableReason}')
        print(f'A: {self.A}')


if __name__ == '__main__':
    species = 10
    patches = 5
    for seed in range(1, 101):
        net = DispersalNetwork(randomSeed=seed, species=species, patches=patches)
        net.compute()
        net.disp()
