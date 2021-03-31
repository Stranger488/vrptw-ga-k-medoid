import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spatiotemporal import Spatiotemporal
from solver import Solver
from pyvrp_solver import PyVRPSolver

from plot import Plot
from utils import Utils

# from config_reduced import *
from config_standard import *


class Kernel:
    def __init__(self):
        self.utils = Utils()
        self.plotter = Plot()

        self.numpy_rand = np.random.RandomState(42)

    def make_solution(self, init_dataset, tws_all, service_time_all, k=None, distance='spatiotemp', plot=False,
                      text=False, output_dir='cluster_result/', eval_method='default'):

        # Init and calculate all spatiotemporal distances
        spatiotemporal = Spatiotemporal(init_dataset, tws_all, service_time_all, k1, k2, k3, alpha1, alpha2)
        spatiotemporal.calculate_all_distances()

        # Reduce depot
        dataset_reduced = init_dataset[1:][:]
        tws_reduced = tws_all[1:]

        spatio_points_dist = np.delete(spatiotemporal.euclidian_dist_all, 0, 0)
        spatio_points_dist = np.delete(spatio_points_dist, 0, 1)

        spatiotemporal_points_dist = np.delete(spatiotemporal.spatiotemporal_dist_all, 0, 0)
        spatiotemporal_points_dist = np.delete(spatiotemporal_points_dist, 0, 1)

        if distance == 'spatiotemp':
            solver = Solver(Z, spatiotemporal_points_dist, P, ng, Pc, Pm, Pmb, k=k, numpy_rand=self.numpy_rand)
        else:
            solver = Solver(Z, spatio_points_dist, P, ng, Pc, Pm, Pmb, k=k, numpy_rand=self.numpy_rand)

        # Result will be an array of clusters, where row is a cluster, value in column - point index
        result = solver.solve()

        # Collect result, making datasets of space data and time windows
        res_dataset = np.array([[dataset_reduced[point] for point in cluster] for cluster in result])
        res_tws = np.array([[tws_reduced[point] for point in cluster] for cluster in result])

        for i, cluster in enumerate(res_dataset):
            # Create coords file
            coord_df = pd.DataFrame(res_dataset[i], columns=['X', 'Y'])

            coord_df.loc[-1] = init_dataset[0]
            coord_df.index = coord_df.index + 1  # shifting index
            coord_df.sort_index(inplace=True)

            coord_df.to_csv('cluster_result/' + output_dir + 'coords{}.txt'.format(i), sep=' ', index=False)

            # Create time parameters file
            tw_df = pd.DataFrame(res_tws[i], columns=['TW_early', 'TW_late'])

            tw_df.loc[-1] = tws_all[0]
            tw_df.index = tw_df.index + 1  # shifting index
            tw_df.sort_index(inplace=True)

            tw_df.insert(2, 'TW_service_time', [service_time_all[i][0] for i in range(len(tw_df))])

            tw_df.to_csv('cluster_result/' + output_dir + 'params{}.txt'.format(i), index=False, sep=' ')

        # Output distance matrix
        distance_df = pd.DataFrame(spatiotemporal.euclidian_dist_all)
        distance_df.to_csv('cluster_result/' + output_dir + 'distance_matrix.txt', sep=' ', index=False, header=False)

        tsptw_solver = PyVRPSolver(method='tsp')
        tsptw_results, plots_data = tsptw_solver.solve_tsp(res_dataset.shape[0], data_dir=output_dir)

        if plot:
            self.plotter.plot_clusters(dataset_reduced, res_dataset, res_tws, spatiotemporal.MAX_TW,
                                       np.array(init_dataset[0]), np.array(tws_all[0]), plots_data, axes_text=distance,
                                       text=text)

        # Evaluate solution
        evaluation = self.utils.evaluate_solution(tsptw_results, eval_method=eval_method)

        return evaluation

    def make_solution_pyvrp(self, points_dataset, tws_all, service_time_all, k=None, plot=False, text=False,
                            output_dir='pyvrp_result/', eval_method='default'):
        pyvrp_solver = PyVRPSolver(method='vrp')
        pyvrp_results, plots_data = pyvrp_solver.solve_vrp(points_dataset, tws_all, service_time_all,
                                                           output_dir=output_dir)

        # Evaluate solution
        evaluation = self.utils.evaluate_solution(pyvrp_results, eval_method=eval_method)

        # Reduce depot
        dataset_reduced = points_dataset[1:][:]
        tws_reduced = tws_all[1:]

        if plot:
            max_TW = max(np.subtract(tws_all[:, 1], tws_all[:, 0]))
            self.plotter.plot_clusters(dataset_reduced, points_dataset, tws_reduced, max_TW,
                                       np.array(points_dataset[0]), np.array(tws_all[0]), plots_data, axes_text='pyvrp',
                                       text=text)

        return evaluation

    def solve(self, filename, distance='spatiotemp', plot=False, k=None, output_dir='cluster_result/', text=False,
              method='cluster', eval_method='default'):
        dataset = pd.read_fwf('data/' + filename)

        points_dataset = np.empty((0, 2))
        tws_all = np.empty((0, 2))
        service_time_all = np.empty((0, 1))

        points_dataset, tws_all, service_time_all = self.utils.read_standard_dataset(dataset, points_dataset, tws_all,
                                                                                     service_time_all)
        if method == 'cluster':
            val = self.make_solution(points_dataset, tws_all, service_time_all, k=int(dataset['VEHICLE_NUMBER'][0]),
                                     distance=distance, plot=plot, output_dir=output_dir, text=text,
                                     eval_method=eval_method)
        elif method == 'pyvrp':
            val = self.make_solution_pyvrp(points_dataset, tws_all, service_time_all, output_dir=output_dir,
                                           eval_method=eval_method)
        else:
            val = None

        return val

    def solve_and_plot(self, datasets):
        st = []
        s = []

        pyvrp = []
        for dataset in datasets:
            print(dataset['name'])
            if dataset['method'] == 'pyvrp':
                pyvrp.append(self.solve(dataset['data_file'], distance='spatial', plot=dataset['plot'],
                                        output_dir=dataset['output_dir'], text=dataset['text'],
                                        method=dataset['method'],
                                        eval_method=dataset['eval_method']))
            else:
                st.append(self.solve(dataset['data_file'], distance='spatiotemp', plot=dataset['plot'],
                                     output_dir=dataset['output_dir'], text=dataset['text'], method=dataset['method'],
                                     eval_method=dataset['eval_method']))
                s.append(self.solve(dataset['data_file'], distance='spatial', plot=dataset['plot'],
                                    output_dir=dataset['output_dir'], text=dataset['text'], method=dataset['method'],
                                    eval_method=dataset['eval_method']))

        for i, dataset in enumerate(datasets):
            if dataset['method'] == 'pyvrp':
                print("Pyvrp res on {}: {}".format(dataset['name'], pyvrp[i]))
            else:
                print("Spatiotemporal res on {}: {}".format(dataset['name'], st[i]))
                print("Spatial res on {}: {}\n".format(dataset['name'], s[i]))

        if True in [d['plot'] for d in datasets]:
            plt.show()
