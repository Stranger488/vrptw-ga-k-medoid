import os
import sys
from multiprocessing import Pool
from time import time

from cluster_solver_parallel import ClusterSolverParallel
from plot import Plot


class LauncherCluster:
    def __init__(self, in_dataset_series=None, proc_count=os.cpu_count(), mode=False):
        self._mode = mode
        self._proc_count = proc_count
        self._in_dataset_series = __import__(in_dataset_series)
        self._testing_datasets = self._in_dataset_series.testing_datasets

        self._plotter = Plot()

        self._BASE_DIR = sys.path[0]

    def launch(self):
        output = open(self._BASE_DIR + '/result/' + 'time_cluster_parallel.csv', 'w')

        if self._mode:
            times = []
            cpus = [i for i in range(1, self._proc_count + 1)]
            for i in cpus:
                t, _ = self._make_solving(i)
                times.append(t)
                output.write('{} {}\n'.format(i, round(t, 4)))

            self._plotter.plot_parallel_time(cpus, times)
            self._plotter.show()
        else:
            t, _ = self._make_solving(self._proc_count)
            output.write('{} {}\n'.format(self._proc_count, round(t, 4)))

        output.close()

    def _make_solving(self, process_count):
        full_result = []
        t1 = time()
        with Pool(process_count) as p:
            cluster_solver = ClusterSolverParallel()
            result = p.map(cluster_solver.solve_and_plot, self._testing_datasets)
            full_result.append(result)
        t2 = time()

        return (t2 - t1), result
