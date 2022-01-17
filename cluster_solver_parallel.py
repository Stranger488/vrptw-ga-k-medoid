import pathlib
import sys

import numpy as np
import pandas as pd

from cluster.cluster_solver import ClusterSolver
from cluster.spatiotemporal import Spatiotemporal
from cluster_config_parallel import *
from plot import Plot


class ClusterSolverParallel:
    def __init__(self):
        self._plotter = Plot()
        self._numpy_rand = np.random.RandomState(42)
        self._BASE_DIR = sys.path[0]

    def solve_and_plot(self, dataset):
        print(dataset['name'])

        self._solve(dataset['data_file'], output_dir=dataset['output_dir'], k3_arg=dataset['k3'])

        print("Spatiotemporal res on {}".format(dataset['name']))

    def _solve(self, filename, k=None, output_dir='cluster_result/', k3_arg=2.0):
        dataset = pd.read_fwf(self._BASE_DIR + '/data/' + filename)

        points_dataset = np.empty((0, 2))
        tws_all = np.empty((0, 2))
        service_time_all = np.empty((0, 1))

        points_dataset, tws_all, service_time_all = self._read_standard_dataset(dataset, points_dataset, tws_all,
                                                                                service_time_all)
        val = self._make_solution(points_dataset, tws_all, service_time_all, k=int(dataset['VEHICLE_NUMBER'][0]),
                                  output_dir=output_dir, k3_arg=k3_arg)
        return val

    def _make_solution(self, init_dataset, tws_all, service_time_all, k=None, output_dir='cluster_result/', k3_arg=2.0):
        pathlib.Path(self._BASE_DIR + '/result/cluster_result/' + output_dir).mkdir(parents=True, exist_ok=True)

        # Init and calculate all spatiotemporal distances
        spatiotemporal = Spatiotemporal(init_dataset, tws_all, service_time_all, k1=k1, k2=k2, k3=k3_arg,
                                        alpha1=alpha1, alpha2=alpha2)
        spatiotemporal.calculate_all_distances()

        # Reduce depot
        dataset_reduced = init_dataset[1:][:]
        tws_reduced = tws_all[1:]
        spatiotemporal_points_dist = np.delete(spatiotemporal.spatiotemporal_dist_all, 0, 0)
        spatiotemporal_points_dist = np.delete(spatiotemporal_points_dist, 0, 1)

        cluster_solver = ClusterSolver(spatiotemporal_points_dist, Z=Z, P=P, ng=ng, Pc=Pc, Pm=Pm, Pmb=Pmb, k=k,
                                       numpy_rand=self._numpy_rand)
        # Result will be an array of clusters, where row is a cluster, value in column - point index
        result = cluster_solver.solve_cluster(output_dir)

        # Collect and parse cluster solution
        res_dataset, res_tws = self._collect_cluster_result(dataset_reduced, tws_reduced, result, init_dataset,
                                                            output_dir, tws_all, service_time_all, spatiotemporal)

        return None

    def _collect_cluster_result(self, dataset_reduced, tws_reduced, result, init_dataset, output_dir, tws_all,
                                service_time_all, spatiotemporal):
        # Collect result, making datasets of space data and time_cluster windows
        res_dataset = np.array([[dataset_reduced[point] for point in cluster] for cluster in result])
        res_tws = np.array([[tws_reduced[point] for point in cluster] for cluster in result])

        for i, cluster in enumerate(res_dataset):
            # Create coords file
            coord_df = pd.DataFrame(res_dataset[i], columns=['X', 'Y'])

            coord_df.loc[-1] = init_dataset[0]
            coord_df.index = coord_df.index + 1  # shifting index
            coord_df.sort_index(inplace=True)

            coord_df.to_csv(self._BASE_DIR + '/result/cluster_result/' + output_dir + 'coords{}.txt'.format(i), sep=' ',
                            index=False)

            # Create time_cluster parameters file
            tw_df = pd.DataFrame(res_tws[i], columns=['TW_early', 'TW_late'])

            tw_df.loc[-1] = tws_all[0]
            tw_df.index = tw_df.index + 1  # shifting index
            tw_df.sort_index(inplace=True)

            tw_df.insert(2, 'TW_service_time', [service_time_all[i][0] for i in range(len(tw_df))])

            tw_df.to_csv(self._BASE_DIR + '/result/cluster_result/' + output_dir + 'params{}.txt'.format(i),
                         index=False,
                         sep=' ')

        # Output distance matrix
        distance_df = pd.DataFrame(spatiotemporal.euclidian_dist_all)
        distance_df.to_csv(self._BASE_DIR + '/result/cluster_result/' + output_dir + 'distance_matrix.txt', sep=' ',
                           index=False, header=False)

        return res_dataset, res_tws

    def _read_standard_dataset(self, dataset, points_dataset, tws_all, service_time_all):
        for i in range(dataset.shape[0]):
            tws_all = np.concatenate((tws_all, [[dataset['READY_TIME'][i],
                                                 dataset['DUE_DATE'][i]]]), axis=0)

            service_time_all = np.concatenate((service_time_all, [[dataset['SERVICE_TIME'][i]]]), axis=0)

            points_dataset = np.concatenate((points_dataset, [[dataset['XCOORD'][i],
                                                               dataset['YCOORD'][i]]]), axis=0)

        return points_dataset, tws_all, service_time_all
