from multiprocessing import Pool

import numpy as np
import pandas as pd
from numpy import ndarray

from src.cluster.cluster_launch_entry import ClusterLaunchEntry
from src.cluster.cluster_result_entry import ClusterResultEntry
from src.cluster.cluster_solver import ClusterSolver
from src.cluster.spatiotemporal import Spatiotemporal
from src.common.plot import Plot
from src.common.utils import timing, create_directory, read_standard_dataset
from src.tsptw.tsptw_launch_entry import TSPTWLaunchEntry
from src.tsptw.tsptw_result_entry import TSPTWResultEntry
from src.tsptw.tsptw_solver import TSPTWSolver
from src.vrptw.vrptw_launch_entry import VRPTWLaunchEntry


# only such functions can be Pickable for Multiprocessing
# functions (built-in and user-defined) accessible from the top level of a module (using def, not lambda);
def lambda_cluster_default(s, t):
    return timing(t)(s.solve)()


class VRPTWSolver:
    def __init__(self, vrptw_launch_entry: VRPTWLaunchEntry):
        self._vrptw_launch_entry = vrptw_launch_entry
        self._cluster_launch_entry_arr = vrptw_launch_entry.cluster_launch_entry_arr
        self._tsptw_launch_entry_arr = vrptw_launch_entry.tsptw_launch_entry_arr

        self._plotter = Plot()

    # Выводить ли график решения - задается в каждом entry
    def solve(self, solve_cluster, solve_tsptw):
        if solve_cluster is not None:
            cluster_result_entry_arr = self._solve_in_cluster_mode(solve_cluster)
        if solve_tsptw is not None:
            tsptw_result_entry_arr = self._solve_in_tsptw_mode(solve_tsptw)

    def _solve_in_cluster_mode(self, solve_cluster):
        full_result = np.array([], dtype=ClusterResultEntry)

        if solve_cluster == 'default':
            return self._solve_cluster_parallel(full_result,
                                                lambda_cluster_default)

        if solve_cluster == 'dm':
            return self._solve_cluster_sequential(full_result,
                                                  lambda s, t: timing(t)(s.solve_cluster_core_data_mining)())

        if solve_cluster == 'sequential':
            return self._solve_cluster_sequential(full_result,
                                                  lambda s, t: timing(t)(s.solve)())

    def _solve_cluster_parallel(self, full_result, lambda_to_solve):
        with Pool(self._vrptw_launch_entry.proc_count) as p:
            args = [(e, lambda_to_solve) for e in self._cluster_launch_entry_arr]
            result = p.starmap(self._solve_cluster_base, args)
            full_result = np.append(full_result, result)
        return full_result

    def _solve_cluster_sequential(self, full_result: ndarray, lambda_to_solve):
        for cluster_launch_entry in self._cluster_launch_entry_arr:
            result = self._solve_cluster_base(cluster_launch_entry, lambda_to_solve)
            full_result = np.append(full_result, result)
        return full_result

    def _solve_cluster_base(self, cluster_launch_entry: ClusterLaunchEntry, lambda_to_solve):
        create_directory(self._vrptw_launch_entry.CLUSTER_OUTPUT + cluster_launch_entry.common_id)
        points_dataset, tws_all, service_time_all, vehicle_number = VRPTWSolver.read_input_for_cluster_mode(
            self._vrptw_launch_entry.BASE_DIR + '/input/task/' + cluster_launch_entry.dataset.data_file)

        dataset_reduced, spatiotemporal, spatiotemporal_points_dist, tws_reduced = self.calculate_spatiotemporal(
            cluster_launch_entry, points_dataset, service_time_all, tws_all)
        cluster_solver = ClusterSolver(distances=spatiotemporal_points_dist, Z=cluster_launch_entry.Z,
                                       P=cluster_launch_entry.P,
                                       ng=cluster_launch_entry.ng, Pc=cluster_launch_entry.Pc,
                                       Pm=cluster_launch_entry.Pm, Pmb=cluster_launch_entry.Pmb,
                                       k=vehicle_number)

        # Result will be an array of clusters, where row is a cluster, value in column - point index
        result = lambda_to_solve(cluster_solver, self._vrptw_launch_entry.CLUSTER_OUTPUT
                                 + cluster_launch_entry.common_id + '/time_cluster.csv')
        # Collect and parse cluster solution
        res_dataset, res_tws = self._collect_cluster_result(dataset_reduced, tws_reduced, result, points_dataset,
                                                            cluster_launch_entry.common_id + '/', tws_all,
                                                            service_time_all, spatiotemporal)
        return ClusterResultEntry(result, res_dataset, res_tws, spatiotemporal, dataset_reduced)

    def _collect_cluster_result(self, dataset_reduced, tws_reduced, result, init_dataset, output_dir, tws_all,
                                service_time_all, spatiotemporal):
        # Collect output, making datasets of space input and time_cluster windows
        res_dataset = [[dataset_reduced[point] for point in cluster if point != -1] for cluster in result]
        res_tws = [[tws_reduced[point] for point in cluster if point != -1] for cluster in result]

        for i, cluster in enumerate(res_dataset):
            # Create coords file
            coord_df = pd.DataFrame(res_dataset[i], columns=['X', 'Y'])

            coord_df.loc[-1] = init_dataset[0]
            coord_df.index = coord_df.index + 1  # shifting index
            coord_df.sort_index(inplace=True)

            coord_df.to_csv(self._vrptw_launch_entry.CLUSTER_OUTPUT + output_dir + 'coords{}.txt'.format(i), sep=' ',
                            index=False)

            # Create time_cluster parameters file
            tw_df = pd.DataFrame(res_tws[i], columns=['TW_early', 'TW_late'])

            tw_df.loc[-1] = tws_all[0]
            tw_df.index = tw_df.index + 1  # shifting index
            tw_df.sort_index(inplace=True)

            tw_df.insert(2, 'TW_service_time', np.array([service_time_all[i][0] for i in range(len(tw_df))]))

            tw_df.to_csv(self._vrptw_launch_entry.CLUSTER_OUTPUT + output_dir + 'params{}.txt'.format(i),
                         index=False,
                         sep=' ')

        # Output distance matrix
        distance_df = pd.DataFrame(spatiotemporal.euclidian_dist_all)
        distance_df.to_csv(self._vrptw_launch_entry.CLUSTER_OUTPUT + output_dir + 'distance_matrix.txt', sep=' ',
                           index=False, header=False)

        return res_dataset, res_tws

    @staticmethod
    def calculate_spatiotemporal(cluster_launch_entry, init_dataset, service_time_all, tws_all):
        # Init and calculate all spatiotemporal distances
        spatiotemporal = Spatiotemporal(init_dataset, tws_all, service_time_all,
                                        k1=cluster_launch_entry.k1, k2=cluster_launch_entry.k2,
                                        k3=cluster_launch_entry.k3,
                                        alpha1=cluster_launch_entry.alpha1, alpha2=cluster_launch_entry.alpha2)
        spatiotemporal.calculate_all_distances()
        # Reduce depot
        dataset_reduced = init_dataset[1:][:]
        tws_reduced = tws_all[1:]
        spatiotemporal_points_dist = np.delete(spatiotemporal.spatiotemporal_dist_all, 0, 0)
        spatiotemporal_points_dist = np.delete(spatiotemporal_points_dist, 0, 1)
        return dataset_reduced, spatiotemporal, spatiotemporal_points_dist, tws_reduced

    def _solve_tsptw_base(self, tsptw_launch_entry: TSPTWLaunchEntry, lambda_to_solve):
        create_directory(self._vrptw_launch_entry.TSPTW_OUTPUT + tsptw_launch_entry.common_id)

        # TODO: read data here and remember necessary for plotting and evaluation

        tsptw_solver = TSPTWSolver(route=tsptw_launch_entry.route,
                                   population_size=tsptw_launch_entry.population_size,
                                   mutation_rate=tsptw_launch_entry.mutation_rate,
                                   elite=tsptw_launch_entry.elite,
                                   generations=tsptw_launch_entry.generations,
                                   pool_size=tsptw_launch_entry.proc_count,
                                   k1=tsptw_launch_entry.k1, k2=tsptw_launch_entry.k2)
        tsptw_results, plots_data = lambda_to_solve(tsptw_solver,
                                                    self._vrptw_launch_entry.TSPTW_OUTPUT
                                                    + tsptw_launch_entry.common_id
                                                    + '/time_tsptw.csv')

        evaluation = self._evaluate_solution(tsptw_results,
                                             self._vrptw_launch_entry.EVALUATION_OUTPUT
                                             + tsptw_launch_entry.common_id
                                             + '/evaluation.csv')

        return TSPTWResultEntry(tsptw_results, plots_data, evaluation)

    def _solve_in_tsptw_mode(self, solve_tsptw):
        full_result = np.array([], dtype=TSPTWResultEntry)

        if solve_tsptw == 'default':
            return self._solve_tsptw_sequential(full_result,
                                                lambda s, t: timing(t)(s.solve_tsptw_parallel)())

        if solve_tsptw == 'dm':
            return self._solve_tsptw_sequential(full_result,
                                                lambda s, t: timing(t)(s.solve_tsptw_core_data_mining)())

        if solve_tsptw == 'sequential':
            return self._solve_tsptw_sequential(full_result,
                                                lambda s, t: timing(t)(s.solve)())

    def _solve_tsptw_sequential(self, full_result, lambda_to_solve):
        for tsptw_launch_entry in self._tsptw_launch_entry_arr:
            result = self._solve_tsptw_base(tsptw_launch_entry, lambda_to_solve)
            full_result = np.append(full_result, result)
        return full_result

    @staticmethod
    def read_input_for_cluster_mode(path):
        dataset = pd.read_fwf(path)
        points_dataset, tws_all, service_time_all = read_standard_dataset(dataset)
        vehicle_number = int(dataset['VEHICLE_NUMBER'][0])
        return points_dataset, tws_all, service_time_all, vehicle_number

    def _evaluate_solution(self, tsptw_results, output_dir):
        create_directory(output_dir)

        total_dist = 0.0
        wait_time = 0.0
        late_time = 0.0

        for result in tsptw_results:
            total_dist += result['Distance'][len(result) - 1]
            wait_time += result['Wait_Time'][len(result) - 1]
            late_time += result['Late_Time'][len(result) - 1]

        evaluation = self._vrptw_launch_entry.c_D * total_dist \
                     + self._vrptw_launch_entry.c_T * wait_time \
                     + self._vrptw_launch_entry.c_L * late_time

        result = pd.DataFrame([total_dist, wait_time, late_time, evaluation])
        result.to_csv(output_dir + 'evaluation.csv', sep=' ',
                      index=False, header=False)

        return evaluation
