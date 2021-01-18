import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d.axis3d import Axis

from spatiotemporal import Spatiotemporal
from solver import Solver

k1 = 1.0
k2 = 1.5
k3 = 2.0

alpha1 = 0.5
alpha2 = 0.5

# Acceptable TSPTW subproblem size Z
Z = 10

# Population size, i. e. number of chromosomes in population
P = 100

# Number of all generations
ng = 100

# Crossover probability
Pc = 0.9

# Mutation probability
Pm = 0.1

# D_MX crossover Built-in mutation probability
Pmb = 0.05

# Plot parameters
figsize_standart = (25, 15)
dpi_standart = 400
linewidth_standart = 0.5
width = depth = 0.5


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

plt.rc('font', size=5)  # controls default text sizes
plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
plt.rc('ytick', labelsize=8)

seed = 0
random.seed(seed)
np.random.seed(seed)


def make_solution(init_dataset, tws_all, service_time_all, k=None, distance='spatiotemp', plot=True, text=False):
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

    # Estimate solution
    tsptw_result = None
    dist, wait_time, late_time = estimate_solution(result, spatiotemporal_points_dist, tsptw_result)
    print("Total distance: {}".format(dist))
    print("---------")

    if plot:
        # Plot data with time windows
        plot_clusters(dataset_reduced, res_dataset, res_tws, spatiotemporal.MAX_TW,
                      np.array(init_dataset[0]), np.array(tws_all[0]))

    return dist + wait_time + late_time


def solve_test(distance='spatiotemp', plot=True, k=None):
    tws_all = np.array([
        [0, 720],
        [60, 120],
        [420, 480],
        [60, 120],
        [420, 480],
        [60, 120],
        [420, 480]
    ])

    service_time_all = np.array([
        0, 10, 10, 10, 10, 10, 10
    ])

    test_dataset = np.array([[50, 50], [10, 10], [30, 10], [30, 30], [70, 70], [70, 90], [90, 90]])

    val = make_solution(test_dataset, tws_all, service_time_all, k=k, distance=distance, plot=plot, text=True)

    return val


def read_standard_dataset(dataset, points_dataset, tws_all, service_time_all):
    for i in range(dataset.shape[0]):
        tws_all = np.concatenate((tws_all, [[dataset['READY_TIME'][i],
                                             dataset['DUE_DATE'][i]]]), axis=0)

        service_time_all = np.concatenate((service_time_all, [[dataset['SERVICE_TIME'][i]]]), axis=0)

        points_dataset = np.concatenate((points_dataset, [[dataset['XCOORD'][i],
                                                           dataset['YCOORD'][i]]]), axis=0)

    return points_dataset, tws_all, service_time_all


def solve(filename, distance='spatiotemp', plot=False, k=None):
    dataset = pd.read_fwf(filename)

    points_dataset = np.empty((0, 2))
    tws_all = np.empty((0, 2))
    service_time_all = np.empty((0, 1))

    points_dataset, tws_all, service_time_all = read_standard_dataset(dataset, points_dataset, tws_all, service_time_all)

    val = make_solution(points_dataset, tws_all, service_time_all, k=int(dataset['VEHICLE_NUMBER'][0]), distance=distance, plot=plot)

    return val


def plot_clusters(init_dataset, dataset, tws, max_tw, depo_spatio, depo_tws, text=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize_standart,
                             dpi=dpi_standart, subplot_kw={'projection': '3d'})

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

    colors = [rgb2hex([np.random.random_sample(), np.random.random_sample(), np.random.random_sample()])
              for _ in dataset]

    for i in range(dataset.shape[0]):
        plot_with_tws(dataset[i], tws[i], max_tw, colors[i], axes)

    axes.scatter(depo_spatio[0], depo_spatio[1], 0.0, c='black', s=1)

    axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[0], c='black', s=1)
    axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[1], c='black', s=1)

    axes.bar3d(depo_spatio[0] - depth / 8., depo_spatio[1] - depth / 8., 0.0, width / 4., depth / 4., max_tw, color='black')

    if text:
        for i, data in enumerate(init_dataset):
            axes.text(data[0], data[1], 0.0, str(i))

    axes.set_zlim(0, None)
    plt.show()


def plot_with_tws(spatial_data, tws, max_tw, colors, axes):
    cluster_size = spatial_data[0].size

    x_data = np.array([i[0] for i in spatial_data])
    y_data = np.array([i[1] for i in spatial_data])

    z_data1 = np.array([i[0] for i in tws])
    z_data2 = np.array([i[1] for i in tws])
    dz_data = np.abs(np.subtract(z_data1, z_data2))

    axes.bar3d(x_data - depth / 8., y_data - depth / 8., 0.0, width / 4., depth / 4., max_tw)
    axes.bar3d(x_data - depth / 2., y_data - depth / 2., z_data1, width, depth, dz_data)

    axes.plot(x_data, y_data, 0.0, linewidth=linewidth_standart)

    axes.scatter(x_data, y_data, 0.0, c=colors, s=cluster_size)
    axes.scatter(x_data, y_data, z_data1, c=colors, s=cluster_size)
    axes.scatter(x_data, y_data, z_data2, c=colors, s=cluster_size)


def estimate_solution(result, spatial_dist, tsptw_result):
    total_dist = 0.0
    wait_time = 0.0
    late_time = 0.0
    for cluster in result:
        for i in range(cluster.size - 1):
            total_dist += spatial_dist[cluster[i]][cluster[i + 1]]

    return total_dist, wait_time, late_time


if __name__ == '__main__':
    # solve_test(k=2)

    r101_reduced_st = solve('data/r101_reduced.txt', distance='spatiotemp')
    r101_reduced_s = solve('data/r101_reduced.txt', distance='spatial')

    # c101_st = solve('data/c101_mod.txt', distance='spatiotemp')
    # c101_s = solve('data/c101_mod.txt', distance='spatial')

    # rc103_st = solve('data/rc103_mod.txt', distance='spatiotemp')
    # rc103_s = solve('data/rc103_mod.txt', distance='spatial')

    print("Spatiotemporal res on r101_reduced: {}".format(r101_reduced_st))
    print("Spatial res on r101_reduced: {}".format(r101_reduced_s))

    # print("Spatiotemporal res on c101: {}".format(c101_st))
    # print("Spatial res on c101: {}".format(c101_s))

    # print("Spatiotemporal res on rc103: {}".format(rc103_st))
    # print("Spatial res on rc103: {}".format(rc103_s))
