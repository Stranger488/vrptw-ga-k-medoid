import os
import sys
from multiprocessing import Pool
from time import time

import pandas as pd

from cluster_solver_parallel import ClusterSolverParallel
from plot import Plot


class LauncherCluster:
    def __init__(self, in_dataset_series=None, proc_count=os.cpu_count(), mode=False, plot_time_stats='full_parallel'):
        self._mode = mode
        self._plot_time_stats = plot_time_stats
        self._proc_count = proc_count
        self._in_dataset_series = __import__(in_dataset_series)
        self._testing_datasets = self._in_dataset_series.testing_datasets

        self._plotter = Plot()

        self._BASE_DIR = sys.path[0]

    def launch(self):
        if self._mode == 'full_parallel':
            output = open(self._BASE_DIR + '/result/' + 'time_cluster_parallel.csv', 'w')
            times = []
            cpus = [i for i in range(1, self._proc_count + 1)]
            for i in cpus:
                t, _ = self._make_solving(i)
                times.append(t)
                output.write('{} {}\n'.format(i, round(t, 4)))
            output.close()
        elif self._mode == 'parallel':
            output = open(self._BASE_DIR + '/result/' + 'time_cluster_parallel.csv', 'w')
            t, _ = self._make_solving(self._proc_count)
            output.write('{} {}\n'.format(self._proc_count, round(t, 4)))
            output.close()

        if self._plot_time_stats:
            cpus_and_times = pd.read_fwf(self._BASE_DIR + '/result/' + 'time_cluster_parallel.csv', header=None).values
            self.make_plot_time(cpus_and_times)

    def _make_solving(self, process_count):
        full_result = []
        t1 = time()
        with Pool(process_count) as p:
            cluster_solver = ClusterSolverParallel()
            result = p.map(cluster_solver.solve_and_plot, self._testing_datasets)
            full_result.append(result)
        t2 = time()

        return (t2 - t1), result

    def make_plot_time(self, cpus_and_times):
        self._plotter.plot_parallel_time(cpus_and_times[:, 0], cpus_and_times[:, 1])
        # self._plotter.show()
