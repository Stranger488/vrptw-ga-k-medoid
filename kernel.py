import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys

from time import time

from cluster.spatiotemporal import Spatiotemporal
from cluster.cluster_solver import ClusterSolver
from tsptw.tsptw_solver import TSPTWSolver

from plot import Plot
from utils import Utils

# from config_reduced import *
from cluster_config_standard import *


class Kernel:
    def __init__(self, k3_from_outer=None):
        if k3_from_outer:
            self.k3 = k3_from_outer

        self.utils = Utils()
        self.plotter = Plot()

        self.numpy_rand = np.random.RandomState(42)

        self.BASE_DIR = sys.path[0]

    def make_solution(self, init_dataset, tws_all, service_time_all, k=None, plot=False,
                      text=False, output_dir='cluster_result/'):
        pathlib.Path(self.BASE_DIR + '/result/cluster_result/' + output_dir).mkdir(parents=True, exist_ok=True)

        # Init and calculate all spatiotemporal distances
        spatiotemporal = Spatiotemporal(init_dataset, tws_all, service_time_all, k1, k2, self.k3, alpha1, alpha2)

        spatiotemporal.calculate_all_distances()

        # Reduce depot
        dataset_reduced = init_dataset[1:][:]
        tws_reduced = tws_all[1:]

        spatiotemporal_points_dist = np.delete(spatiotemporal.spatiotemporal_dist_all, 0, 0)
        spatiotemporal_points_dist = np.delete(spatiotemporal_points_dist, 0, 1)

        solver = ClusterSolver(Z, spatiotemporal_points_dist, P, ng, Pc, Pm, Pmb, k=k, numpy_rand=self.numpy_rand)
        # Result will be an array of clusters, where row is a cluster, value in column - point index
        ts = time()
        result = solver.solve()
        te = time()

        output = open(self.BASE_DIR + '/result/cluster_result/' + output_dir + 'time_cluster.csv', 'w')
        output.write('{}\n'.format(round(te - ts, 4)))
        output.close()

        # Collect result, making datasets of space data and time_cluster windows
        res_dataset = np.array([[dataset_reduced[point] for point in cluster] for cluster in result])
        res_tws = np.array([[tws_reduced[point] for point in cluster] for cluster in result])

        for i, cluster in enumerate(res_dataset):
            # Create coords file
            coord_df = pd.DataFrame(res_dataset[i], columns=['X', 'Y'])

            coord_df.loc[-1] = init_dataset[0]
            coord_df.index = coord_df.index + 1  # shifting index
            coord_df.sort_index(inplace=True)

            coord_df.to_csv(self.BASE_DIR + '/result/cluster_result/' + output_dir + 'coords{}.txt'.format(i), sep=' ', index=False)

            # Create time_cluster parameters file
            tw_df = pd.DataFrame(res_tws[i], columns=['TW_early', 'TW_late'])

            tw_df.loc[-1] = tws_all[0]
            tw_df.index = tw_df.index + 1  # shifting index
            tw_df.sort_index(inplace=True)

            tw_df.insert(2, 'TW_service_time', [service_time_all[i][0] for i in range(len(tw_df))])

            tw_df.to_csv(self.BASE_DIR + '/result/cluster_result/' + output_dir + 'params{}.txt'.format(i), index=False, sep=' ')

        # Output distance matrix
        distance_df = pd.DataFrame(spatiotemporal.euclidian_dist_all)
        distance_df.to_csv(self.BASE_DIR + '/result/cluster_result/' + output_dir + 'distance_matrix.txt', sep=' ', index=False, header=False)

        tsptw_solver = TSPTWSolver()

        ts = time()
        tsptw_results, plots_data = tsptw_solver.solve_tsp(res_dataset.shape[0], data_dir=output_dir)
        te = time()

        output = open(self.BASE_DIR + '/result/tsptw_result/' + output_dir + 'time_tsp.csv', 'w')
        output.write('{}\n'.format(round(te - ts, 4)))
        output.close()

        # Evaluate solution
        evaluation = self.utils.evaluate_solution(tsptw_results, output_dir)

        if plot:
            self.plotter.plot_clusters(dataset_reduced, res_dataset, res_tws, spatiotemporal.MAX_TW,
                                       np.array(init_dataset[0]), np.array(tws_all[0]), plots_data,
                                       text=text)

        return evaluation

    def solve(self, filename, plot=False, k=None, output_dir='cluster_result/', text=False):
        dataset = pd.read_fwf(self.BASE_DIR + '/data/' + filename)

        points_dataset = np.empty((0, 2))
        tws_all = np.empty((0, 2))
        service_time_all = np.empty((0, 1))

        points_dataset, tws_all, service_time_all = self.utils.read_standard_dataset(dataset, points_dataset, tws_all,
                                                                                     service_time_all)
        val = self.make_solution(points_dataset, tws_all, service_time_all, k=int(dataset['VEHICLE_NUMBER'][0]),
                                 plot=plot, output_dir=output_dir, text=text)
        return val

    def solve_and_plot(self, datasets):
        st = []
        for dataset in datasets:
            print(dataset['name'])
            st.append(self.solve(dataset['data_file'], plot=dataset['plot'],
                                 output_dir=dataset['output_dir'], text=dataset['text']))

        for i, dataset in enumerate(datasets):
            print("Spatiotemporal res on {}: {}".format(dataset['name'], st[i]))

        if True in [d['plot'] for d in datasets]:
            plt.show()
