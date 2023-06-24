import numpy as np


class Network:
    def __init__(self, species, randomSeed):
        self.S = species  # numbers of species
        self.seed = randomSeed
        self.A, self.b = None, None
        self.Xf = None  # fixed point
        self.J = None  # Jacobian matrix
        self.eigval = None  # eigenvalues of J
        self.stable = True  # stability of the network

    def spawn_A_b(self):
        """
        randomly spawn A, b\n
        A: normal distribution N(0, 1)\n
        b: (1, 1, ..., 1)
        :return:
        """
        np.random.seed(self.seed)
        self.A = np.random.normal(0, 1, size=(self.S, self.S))
        np.random.seed(self.seed)
        self.b = -1 * np.ones((self.S, 1))

    def solveX(self):
        """
        solve AX=b to find the fixed point\n
        if some N's are negative, regard the point as unstable
        :return:
        """
        self.Xf = np.linalg.solve(self.A, self.b)
        if sum(self.Xf > 0) < self.S:
            self.stable = False  # persistent

    def calc_jacobian(self):
        """
        calculate the Jacobian matrix at the fixed point
        :return:
        """
        temp = self.Xf.reshape(self.S)
        self.J = np.matmul(np.diag(temp), self.A)

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

    def compute(self):
        """
        compute the stability of this network
        :return:
        """
        self.spawn_A_b()
        self.solveX()
        if self.stable:
            self.calc_jacobian()
            self.calc_eigenvalue()

    def disp(self):
        """
        display network information
        :return:
        """
        print(f'Num. Species:{self.S}')
        print(f'Seed: {self.seed}')
        print(f'Fixed point: {self.Xf}')
        print(f'Eigenvalues: {self.eigval}')
        print(f'Stable: {self.stable}')


if __name__ == '__main__':
    species = 10
    for seed in range(1, 11):
        net = Network(species=species, randomSeed=seed)
        net.compute()
        # if not net.stable:
        #     continue
        net.disp()
