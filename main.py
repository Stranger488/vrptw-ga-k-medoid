import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d.axis3d import Axis

from libs.pyVRP import plot_tour_coordinates
from pyvrp_solver import PyVRPSolver
from spatiotemporal import Spatiotemporal
from solver import Solver

from config_reduced import *
# from config_standard import *


# fix wrong z-offsets in 3d plot
def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0, 0, 1.0 / 4]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs


if not hasattr(Axis, "_get_coord_info_old"):
    Axis._get_coord_info_old = Axis._get_coord_info
Axis._get_coord_info = _get_coord_info_new

seed = 0
random.seed(seed)
np.random.seed(seed)


def make_solution(init_dataset, tws_all, service_time_all, k=None, distance='spatiotemp', plot=False, text=False,
                  output_dir='cluster_result/'):
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
        solver = Solver(Z, spatiotemporal_points_dist, P, ng, Pc, Pm, Pmb, k=k)
    else:
        solver = Solver(Z, spatio_points_dist, P, ng, Pc, Pm, Pmb, k=k)

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

    # tsptw_solver = PyVRPSolver()
    # tsptw_results, plots_data = tsptw_solver.solve(res_dataset.shape[0], data_dir=output_dir)

    if plot:
        # plot_clusters(dataset_reduced, res_dataset, res_tws, spatiotemporal.MAX_TW,
        #               np.array(init_dataset[0]), np.array(tws_all[0]), plots_data, axes_text=distance)
        plot_clusters(dataset_reduced, res_dataset, res_tws, spatiotemporal.MAX_TW,
                      np.array(init_dataset[0]), np.array(tws_all[0]), None, axes_text=distance)

    # Evaluate solution
    # evaluation = evaluate_solution(tsptw_results)

    # return evaluation


def read_standard_dataset(dataset, points_dataset, tws_all, service_time_all):
    for i in range(dataset.shape[0]):
        tws_all = np.concatenate((tws_all, [[dataset['READY_TIME'][i],
                                             dataset['DUE_DATE'][i]]]), axis=0)

        service_time_all = np.concatenate((service_time_all, [[dataset['SERVICE_TIME'][i]]]), axis=0)

        points_dataset = np.concatenate((points_dataset, [[dataset['XCOORD'][i],
                                                           dataset['YCOORD'][i]]]), axis=0)

    return points_dataset, tws_all, service_time_all


def solve(filename, distance='spatiotemp', plot=False, k=None, output_dir='cluster_result/'):
    dataset = pd.read_fwf('data/' + filename)

    points_dataset = np.empty((0, 2))
    tws_all = np.empty((0, 2))
    service_time_all = np.empty((0, 1))

    points_dataset, tws_all, service_time_all = read_standard_dataset(dataset, points_dataset, tws_all,
                                                                      service_time_all)

    # val = make_solution(points_dataset, tws_all, service_time_all, k=int(dataset['VEHICLE_NUMBER'][0]),
    #                     distance=distance, plot=plot, output_dir=output_dir)
    make_solution(points_dataset, tws_all, service_time_all, k=int(dataset['VEHICLE_NUMBER'][0]),
                        distance=distance, plot=plot, output_dir=output_dir)

    # return val


def plot_clusters(init_dataset, dataset, tws, max_tw, depo_spatio, depo_tws, plots_data, axes_text=None):
    plt.rc('font', size=5)  # controls default text sizes
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize_standart,
                             dpi=dpi_standart, subplot_kw={'projection': '3d'})

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

    axes.set_title(axes_text)

    colors = [rgb2hex([np.random.random_sample(), np.random.random_sample(), np.random.random_sample()])
              for _ in dataset]

    for i in range(dataset.shape[0]):
        plot_with_tws(dataset[i], tws[i], max_tw, colors[i], axes)
        # plot_tour_coordinates(plots_data[i]['coordinates'], plots_data[i]['ga_vrp'], axes, colors[i],
        #                       n_depots=plots_data[i]['n_depots'], route=plots_data[i]['route'])

    axes.scatter(depo_spatio[0], depo_spatio[1], 0.0, c='black', s=1)

    axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[0], c='black', s=1)
    axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[1], c='black', s=1)

    axes.bar3d(depo_spatio[0] - depth / 8., depo_spatio[1] - depth / 8., 0.0, width / 4., depth / 4., max_tw,
               color='black')

    for i, data in enumerate(init_dataset):
        axes.text(data[0], data[1], 0.0, str(i + 1))

    axes.set_zlim(0, None)


def plot_with_tws(spatial_data, tws, max_tw, colors, axes):
    cluster_size = spatial_data[0].size

    x_data = np.array([i[0] for i in spatial_data])
    y_data = np.array([i[1] for i in spatial_data])

    z_data1 = np.array([i[0] for i in tws])
    z_data2 = np.array([i[1] for i in tws])
    dz_data = np.abs(np.subtract(z_data1, z_data2))

    axes.bar3d(x_data - depth / 8., y_data - depth / 8., 0.0, width / 4., depth / 4., max_tw)
    axes.bar3d(x_data - depth / 2., y_data - depth / 2., z_data1, width, depth, dz_data)

    axes.scatter(x_data, y_data, 0.0, c=colors, s=cluster_size)
    axes.scatter(x_data, y_data, z_data1, c=colors, s=cluster_size)
    axes.scatter(x_data, y_data, z_data2, c=colors, s=cluster_size)


def evaluate_solution(tsptw_results):
    total_dist = 0.0
    wait_time = 0.0
    late_time = 0.0

    for result in tsptw_results:
        total_dist += result['Distance'][len(result) - 1]
        wait_time += result['Wait_Time'][len(result) - 1]
        late_time += result['Late_Time'][len(result) - 1]

    return c_D * total_dist + c_T * wait_time + c_L * late_time


def solve_and_plot(datasets):
    st = []
    s = []
    for dataset in datasets:
        st.append(solve(dataset['data_file'], distance='spatiotemp', plot=dataset['plot'], output_dir=dataset['output_dir']))
        s.append(solve(dataset['data_file'], distance='spatial', plot=dataset['plot'], output_dir=dataset['output_dir']))

    for i, dataset in enumerate(datasets):
        print("Spatiotemporal res on {}: {}".format(dataset['name'], st[i]))
        print("Spatial res on {}: {}\n".format(dataset['name'], s[i]))

    if True in [d['plot'] for d in datasets]:
        plt.show()


if __name__ == '__main__':
    test_dataset = {
        'data_file': 'test.txt',
        'output_dir': 'test_output/',
        'plot': True,
        'name': 'test'
    }

    r101_reduced_dataset = {
        'data_file': 'r101_reduced.txt',
        'output_dir': 'r101_reduced_output/',
        'plot': True,
        'name': 'r101_reduced'
    }

    c101_dataset = {
        'data_file': 'c101_mod.txt',
        'output_dir': 'c101_output/',
        'plot': False,
        'name': 'c101'
    }

    r101_dataset = {
        'data_file': 'r101_mod.txt',
        'output_dir': 'r101_output/',
        'plot': False,
        'name': 'r101'
    }

    rc101_dataset = {
        'data_file': 'rc101_mod.txt',
        'output_dir': 'rc101_output/',
        'plot': False,
        'name': 'rc101'
    }

    c109_dataset = {
        'data_file': 'c109_mod.txt',
        'output_dir': 'c109_output/',
        'plot': False,
        'name': 'c109'
    }

    r109_dataset = {
        'data_file': 'r109_mod.txt',
        'output_dir': 'r109_output/',
        'plot': False,
        'name': 'r109'
    }

    rc103_dataset = {
        'data_file': 'rc103_mod.txt',
        'output_dir': 'rc103_output/',
        'plot': False,
        'name': 'rc103'
    }

    solve_and_plot([test_dataset, ])

    # solve_and_plot([r101_reduced_dataset, ])

    # solve_and_plot([c101_dataset, r101_dataset, rc101_dataset, c109_dataset, r109_dataset, rc103_dataset])

