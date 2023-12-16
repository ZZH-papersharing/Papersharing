import multiprocessing as mp
import os
import time
import warnings
from tkinter import filedialog

import pandas as pd

from compute import *


class NetworkManager:
    """
    This class works as an interface to compute networks.
    """

    def __init__(self, species=40, patches=5, randomSeed=81, weightSeed=0, record=True,
                 initial='random', growth=5, n0=5, alpha0=0.2, N0=None, Alpha0=None,
                 miu_n0=1, sgm_n0=0.2, miu_alpha=0.5, sgm_alpha=0.01, scr=0,
                 connectance=0.6, sgm_aij=0.1, sgm_qx=10, rho=0.5, Adiag=1, dispersal=10, kappa=2,
                 method_ode=2, method_alpha='Softmax', dt=1e-3, maxIteration=100e4, maxError=1e-4,
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
        # p = max(2, np.log10((dispersal+1)*(kappa+1)))
        # dt = 1 / 10 ** np.ceil(p)
        # print(p, dt)
        self.config = {NetParam.randomSeed: randomSeed,
                       NetParam.weightSeed: weightSeed,
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
                       NetParam.scr: scr,
                       NetParam.record: record,

                       NetParam.method_alpha: method_alpha,
                       NetParam.method_ode: method_ode,
                       NetParam.dt: dt,
                       NetParam.maxIteration: maxIteration,
                       NetParam.maxError: maxError,
                       }

        self.net: RandomNetwork | None = None

        self.randos = [0, 0, 0, 0]  # seed, m, sgm, c

    def net2csv(self):
        directory = filedialog.askdirectory()
        self.net = RandomNetwork(self.config)
        np.savetxt(directory + 'net.csv', self.net.A, delimiter=',')

    def computeNet(self):
        """
        Compute the network and plot figures.
        :return:
        """
        self.net = RandomNetwork(self.config)
        cpt = Computer(self.net)
        cpt.compute()
        if self.config[NetParam.record]:
            self.net.plotAll()

    def changeW(self, wseeds, research):
        directory = f"./log/sweep/LinearStability/{research}/"
        directory += (f'S{self.config[NetParam.species]}n{self.config[NetParam.patches]}'
                      f'c{self.config[NetParam.connectance]}rho{self.config[NetParam.rho]}'
                      f'sgm{self.config[NetParam.sgm_aij]}cseed{self.config[NetParam.randomSeed]}')
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        converge = True
        errlog = []
        maxEigvals = []
        kpa_lst = [0, 0.1, 1, 2.5, 5, 7.5, 10, 15]
        for wseed in range(wseeds):
            temp = []
            self.config[NetParam.weightSeed] = wseed
            for kpa in kpa_lst:
                self.config[NetParam.kappa] = kpa
                self.computeNet()
                temp.append(self.net.maxEigval)
                if (not self.net.pst) | (not self.net.stable):
                    converge = False
                    errlog.append({'connectSeed': self.net.seed,
                                   'weightSeed': self.net.weightSeed,
                                   'disperal': self.net.d,
                                   'kappa': self.net.kappa})
            maxEigvals.append(temp)

        df = pd.DataFrame(maxEigvals, columns=kpa_lst).transpose()
        df.columns = map(lambda x: f'wseed={x}', range(wseeds))
        path = directory + f'/d{self.net.d}kpa{kpa_lst[0]}~{kpa_lst[-1]}wseed0~{wseeds}'
        df.to_csv(path + '.csv')

        with open(file=path + 'runlog.txt', mode='w') as f:
            f.write(f'all converged: {converge}')
            f.write(f'errors: {errlog}')

    def changeSign_perStep(self, research):
        directory = f"./log/sweep/LinearStability/{research}/"
        directory += (f'S{self.config[NetParam.species]}n{self.config[NetParam.patches]}'
                      f'c{self.config[NetParam.connectance]}rho{self.config[NetParam.rho]}'
                      f'sgm{self.config[NetParam.sgm_aij]}cseed{self.config[NetParam.randomSeed]}')
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        kpa_lst = [0, 0.1, 1, 2.5, 5, 7.5, 10, 15]
        scr_lst = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
        wseeds = 50
        path = directory + f'/d{self.config[NetParam.dispersal]}kpa{kpa_lst[0]}~{kpa_lst[-1]}scr0~1, seeds{wseeds}'
        if not os.path.exists(path + '.xlsx'):
            with pd.ExcelWriter(path + '.xlsx', mode='w') as writer:
                df = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row 1', 'row 2'], columns=['col 1', 'col 2'])
                df.to_excel(writer, sheet_name='sheet0')
        converge = True
        errlog = []
        for kpa in kpa_lst:
            self.config[NetParam.kappa] = kpa
            maxEigvals, cvg_lst = [], []
            for scr in scr_lst:
                self.config[NetParam.scr] = scr
                temp_eig, temp_cvg = [], []
                for wseed in range(wseeds):
                    self.config[NetParam.weightSeed] = wseed
                    self.computeNet()
                    temp_eig.append(self.net.maxEigval)
                    if (not self.net.pst) | (not self.net.stable):
                        converge = False
                        errlog.append({'connectSeed': self.net.seed,
                                       'weightSeed': self.net.weightSeed,
                                       'disperal': self.net.d,
                                       'kappa': self.net.kappa})
                        temp_cvg.append(False)
                    else:
                        temp_cvg.append(True)
                maxEigvals.append(temp_eig)
                cvg_lst.append(temp_cvg)

            df_eigval = pd.DataFrame(maxEigvals, columns=map(lambda x: f'wseed{x}', range(wseeds))).transpose()
            df_cvg = pd.DataFrame(cvg_lst).transpose()
            df_eigval.columns = map(lambda x: f'scr={x}', scr_lst)
            df_cvg.columns = map(lambda x: f'scr={x}', scr_lst)

            with pd.ExcelWriter(path + '.xlsx', mode='a', if_sheet_exists='overlay') as writer:
                df_eigval.to_excel(writer, sheet_name=f'kpa{kpa}')
                df_cvg.to_excel(writer, sheet_name=f'kpa{kpa}', startcol=2 + len(scr_lst))

        with open(file=path + ' runlog.txt', mode='w') as f:
            f.write(f'all converged: {converge}')
            f.write(f'errors: {errlog}')

    def persistence(self, seeds, research):
        directory = f"./log/sweep/persistence/{research}/"
        directory += (f'S{self.config[NetParam.species]}n{self.config[NetParam.patches]}'
                      f'c{self.config[NetParam.connectance]}rho{self.config[NetParam.rho]}'
                      f'sgm{self.config[NetParam.sgm_aij]},{seeds}seeds')
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # kpa_lst = [0, 0.1, 1, 2.5, 5, 7.5, 10, 15]
        kpa_lst = [0, 15]
        # d_lst = [0.001, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1, 3, 10]
        d_lst = [0.1, 0.5]
        path = directory + f'/d{d_lst[0]}~{d_lst[-1]}kpa{kpa_lst[0]}~{kpa_lst[-1]}'
        if not os.path.exists(path + '.xlsx'):
            with pd.ExcelWriter(path + '.xlsx', mode='w') as writer:
                df = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row 1', 'row 2'], columns=['col 1', 'col 2'])
                df.to_excel(writer, sheet_name='sheet0')

        # detail_data = []
        pst_data = []
        fails = []
        for d in d_lst:
            print('d:', d)
            detail_at_d = []
            pst_at_d = []
            self.config[NetParam.dispersal] = d
            for kpa in kpa_lst:
                print('kpa:', kpa)
                pst_lst = []
                self.config[NetParam.kappa] = kpa
                for seed in range(seeds):
                    para = []
                    np.random.seed(seed)
                    self.config[NetParam.randomSeed] = seed
                    para.append(seed)
                    m = np.random.uniform(0, 10)
                    self.config[NetParam.growth] = m
                    self.config[NetParam.n0] = m
                    para.append(m)
                    # rho = np.random.uniform(0.01, 1)
                    # self.config[NetParam.rho] = rho
                    # para.append(rho)
                    # sgm_aij = np.random.uniform(0.01, 0.35)
                    # self.config[NetParam.sgm_aij] = sgm_aij
                    # para.append(sgm_aij)
                    print(para)
                    try:
                        self.computeNet()
                        if self.net.pst:
                            pst_lst.append(1)
                        else:
                            pst_lst.append(0)
                    except:
                        pst_lst.append(2)
                        # fails.append({'m': m, 'param': change.value['str'], 'value': value, 'seed': seed})
                        pass

                detail_at_d.append(pst_lst)
                pst_at_d.append(pst_lst.count(1) / (pst_lst.count(1) + pst_lst.count(0)))
            detail_df = pd.DataFrame(detail_at_d, columns=range(seeds)).transpose()
            detail_df.columns = map(lambda x: f'kpa={x}', kpa_lst)
            # detail_data.append(detail_df)
            pst_data.append(pst_at_d)
            with pd.ExcelWriter(path + '.xlsx', mode='a', if_sheet_exists='overlay') as writer:
                detail_df.to_excel(writer, sheet_name=f'd{d}')
                # for idx, df in enumerate(detail_data):
                #     df.to_excel(writer, sheet_name=f'd{d_lst[idx]}')
                # pst_df.to_excel(writer, sheet_name='Average')
                # detail_data.to_excel(directory+f'd{self.config[NetParam.dispersal]}.xlsx')
        with pd.ExcelWriter(path + '.xlsx', mode='a', if_sheet_exists='overlay') as writer:
            pst_df_vs_d = pd.DataFrame(pst_data, index=d_lst, columns=map(lambda x: f'kpa={x}', kpa_lst))
            pst_df_vs_d.to_excel(writer, sheet_name='Average_vs_d')
            # pst_df_vs_kpa = pd.DataFrame(pst_data, index=kpa_lst, columns=map(lambda x: f'd={x}',d_lst))
            pst_df_vs_kpa = pst_df_vs_d.transpose()
            pst_df_vs_kpa.index, pst_df_vs_kpa.columns = kpa_lst, map(lambda x: f'd={x}', d_lst)
            pst_df_vs_kpa.to_excel(writer, sheet_name='Average_vs_kpa')

    def for_pst_parallel(self, seed):
        if self.randos[0]:
            np.random.seed(seed)
            self.config[NetParam.randomSeed] = seed

        if self.randos[1]:
            m = np.random.uniform(0, 10)
            self.config[NetParam.growth] = m
            self.config[NetParam.n0] = m

        if self.randos[2]:
            sgm_aij = np.random.uniform(0.01, 0.2)
            self.config[NetParam.sgm_aij] = sgm_aij

        if self.randos[3]:
            c = np.random.uniform(0, 1)
            self.config[NetParam.connectance] = c

        warnings.filterwarnings('error')
        start = time.time()
        try:
            self.computeNet()
            if self.net.pst:
                pst = 1
            else:
                pst = 0
        except (RuntimeWarning, Exception):
            pst = 2
        end = time.time()
        return pst, round(end - start, 2)

    def persistence_parallel(self, seeds, randos: list, kpa_lst=[], d_lst=[]):
        self.randos = randos
        research, allrando = '', ['seed,', 'm,', 'sgm,', 'c,']
        for i in range(len(randos)):
            if randos[i]:
                research += allrando[i]

        directory = f"./log/sweep/persistence/{research}/"
        directory += (f'S{self.config[NetParam.species]}n{self.config[NetParam.patches]}'
                      f'c{self.config[NetParam.connectance]}rho{self.config[NetParam.rho]}'
                      f'sgm{self.config[NetParam.sgm_aij]},{seeds}seeds')
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # kpa_lst = [0, 1, 3, 5, 8, 12, 15]
        # # kpa_lst = [0, 5, 15]
        # d_lst = [0.001, 0.01, 0.1, 1, 10]
        # # d_lst = [1000]
        path = directory + f'/d{d_lst[0]}~{d_lst[-1]}kpa{kpa_lst[0]}~{kpa_lst[-1]}'
        if not os.path.exists(path + '.xlsx'):
            with pd.ExcelWriter(path + '.xlsx', mode='w') as writer:
                df = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row 1', 'row 2'], columns=['col 1', 'col 2'])
                df.to_excel(writer, sheet_name='sheet0')

        pst_data = []
        self.config[NetParam.record] = False
        for d in d_lst:
            print('d:', d)
            detail_at_d = []
            time_at_d = []
            pst_at_d = []
            self.config[NetParam.dispersal] = d
            for kpa in kpa_lst:
                print('kpa:', kpa)
                self.config[NetParam.kappa] = kpa

                pool = mp.Pool(16)
                result_lst = pool.map(self.for_pst_parallel, range(seeds))
                pst_lst = [item[0] for item in result_lst]
                time_lst = [item[1] for item in result_lst]

                detail_at_d.append(pst_lst)
                time_at_d.append(time_lst)
                try:
                    pst_at_d.append(pst_lst.count(1) / (pst_lst.count(1) + pst_lst.count(0)))
                except ZeroDivisionError:
                    pst_at_d.append(0)

            detail_df = pd.DataFrame(detail_at_d, columns=range(seeds)).transpose()
            detail_df.columns = map(lambda x: f'kpa={x}', kpa_lst)
            time_df = pd.DataFrame(time_at_d, columns=range(seeds)).transpose()
            time_df.columns = map(lambda x: f'kpa={x}', kpa_lst)
            pst_data.append(pst_at_d)
            with pd.ExcelWriter(path + '.xlsx', mode='a', if_sheet_exists='overlay') as writer:
                detail_df.to_excel(writer, sheet_name=f'd{d}')
                time_df.to_excel(writer, sheet_name=f'd{d}_time')

        with pd.ExcelWriter(path + '.xlsx', mode='a', if_sheet_exists='overlay') as writer:
            pst_df_vs_d = pd.DataFrame(pst_data, index=d_lst, columns=map(lambda x: f'kpa={x}', kpa_lst))
            pst_df_vs_d.to_excel(writer, sheet_name='Average_vs_d')

            pst_df_vs_kpa = pst_df_vs_d.transpose()
            pst_df_vs_kpa.index, pst_df_vs_kpa.columns = kpa_lst, map(lambda x: f'd={x}', d_lst)
            pst_df_vs_kpa.to_excel(writer, sheet_name='Average_vs_kpa')


if __name__ == '__main__':
    net_manager = NetworkManager()
    # net_manager.net2csv()
    net_manager.computeNet()
    # net_manager.changeW(wseeds=1, research='stvskpa')
    # net_manager.changeW(wseeds=10, research='fixCchangeW')
    # net_manager.changeW(wseeds=10, research='fixWchangeSign,v1')
    # net_manager.changeW(wseeds=10, research='fixSIGNchangeAbsW')
    # net_manager.changeW(wseeds=10, research='fixWstd,changeOffset')
    # net_manager.changeSign_perStep(research='changeSignStep-by-Step')
    # net_manager.persistence(10, research='seed,m')
    # net_manager = NetworkManager(dispersal=0.5)
    # net_manager.persistence(100)
    # net_manager = NetworkManager(rho=0.1)
    # net_manager.persistence_parallel(10, randos=[1, 0, 0, 0])

    # net_manager = NetworkManager(species=40, patches=5, connectance=0.6, rho=0.1, sgm_aij=0.1)
    # net_manager.persistence_parallel(100, randos=[1, 0, 0, 0],
    #                                  kpa_lst=[0, 1, 2, 3, 5, 8, 12, 15], d_lst=[0.001, 0.01, 0.1, 1, 10, 100])
    # net_manager = NetworkManager(species=40, patches=5, connectance=0.6, rho=0.5, sgm_aij=0.1)
    # net_manager.persistence_parallel(100, randos=[1, 0, 0, 0],
    #                                  kpa_lst=[0, 1, 2, 3, 5, 8, 12, 15], d_lst=[0.001, 0.01, 0.1, 1, 10, 100])
    # net_manager = NetworkManager(species=40, patches=5, connectance=0.6, rho=0.9, sgm_aij=0.1)
    # net_manager.persistence_parallel(100, randos=[1, 0, 0, 0],
    #                                  kpa_lst=[0, 1, 2, 3, 5, 8, 12, 15], d_lst=[0.001, 0.01, 0.1, 1, 10, 100])

    #
    # net_manager = NetworkManager(rho=0.1)
    # net_manager.persistence_parallel(100, randos=[1, 1, 0, 0])
    # net_manager = NetworkManager(rho=0.9)
    # net_manager.persistence_parallel(100, randos=[1, 1, 0, 0])
    #
    # net_manager = NetworkManager(rho=0.1)
    # net_manager.persistence_parallel(100, randos=[1, 1, 1, 0])
    # net_manager = NetworkManager(rho=0.9)
    # net_manager.persistence_parallel(100, randos=[1, 1, 1, 0])
    #
    # net_manager = NetworkManager(rho=0.1)
    # net_manager.persistence_parallel(100, randos=[1, 1, 1, 1])
    # net_manager = NetworkManager(species=30, patches=5, connectance=0.6, rho=0.9, sgm_aij=0.1)
    # net_manager.persistence_parallel(100, randos=[1, 1, 1, 1],
    #                                  kpa_lst=[0, 1, 3, 5, 8, 12, 15], d_lst=[100])

    # net_manager = NetworkManager(species=30, patches=5, connectance=0.6, rho=0.9, sgm_aij=0.1, dt=1e-5)
    # net_manager.persistence_parallel(100, randos=[1, 1, 1, 1],
    #                                  kpa_lst=[0, 1, 3, 5, 8, 12, 15], d_lst=[1000])