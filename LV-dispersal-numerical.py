import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class DispersalNetwork:
    def __init__(self, randomSeed, species, patches, dispersal=1, dt=1e-2, maxIteration=1e3, maxError=1e-4):
        self.seed = randomSeed
        np.random.seed(self.seed)

        self.S = species  # numbers of species
        self.n = patches  # numbers of patches
        self.c = 0.2  # connectence
        self.d = dispersal  # dispersal rate
        self.N0 = None  # initial N
        self.dt = dt  # time interval
        self.maxIter = maxIteration  # max iteration time
        self.maxError = maxError  # max error in iteration

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

        self._spawn_A()

        # use 1-D matrix
        self.M = np.ones(self.S * self.n)
        self.N0 = 10 * np.ones(self.S * self.n)

        self._spawn_D()

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
                    if np.random.random() < self.c:
                        A_std[i, j] = np.random.normal(0, 0.25)

        lst_A = []
        for i in range(self.n):
            same = np.random.normal(1, 0.1, size=(self.S, self.S))
            lst_A.append(np.multiply(same, A_std))

        self.A = sp.linalg.block_diag(*lst_A)

    def _spawn_D(self):
        """
        spawn the matrix D
        :return:
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
        N_new = self.RK4(N_old, self.dt)
        # if all elements in |N_new - N_old| are < maxError, regard the point as a fixed point
        iteration = 0
        while True: #sum(abs(N_new - N_old) > self.maxError) > 0:
            temp = N_new
            N_new = self.RK4(N_old, self.dt)
            N_old = temp
            if iteration % 10 == 0:
                self.xlst.append(iteration)
                self.N_lst.append(N_new[0])
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
        # self.calc_jacobian()
        # self.calc_eigenvalue()

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

    def ploteigval(self):
        """
        plot the eigenvalues of Jacobian matrix on the complex plane
        :return:
        """
        x = self.eigval.real
        y = self.eigval.imag

        plt.scatter(x, y)
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.show()

    def plotN(self):

        plt.scatter(self.xlst, self.N_lst)
        plt.xlabel('iteration')
        plt.ylabel('N')
        plt.show()


if __name__ == '__main__':
    species = 10
    patches = 5
    for seed in range(1, 101):
        net = DispersalNetwork(randomSeed=seed, species=species, patches=patches)
        net.compute()
        # if not net.stable:
        #     continue
        net.disp()
        net.plotN()
        # net.ploteigval()
