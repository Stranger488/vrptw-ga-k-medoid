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
Z = 6

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


def estimate_solution(result, spatial_dist, ):
    total_dist = 0.0
    wait_time = 0.0
    late_time = 0.0
    for cluster in result:
        for i in range(cluster.size - 1):
            total_dist += spatial_dist[cluster[i]][cluster[i + 1]]

    return total_dist, wait_time, late_time


def solve(init_dataset, tws_all, service_time_all, k=None, distance='spatiotemp', plot=True):
    # init and calculate all spatiotemporal distances
    spatiotemporal = Spatiotemporal(init_dataset, tws_all, service_time_all, k1, k2, k3, alpha1, alpha2)
    spatiotemporal.calculate_all_distances()

    test_dataset_points = init_dataset[1:][:]
    tws_reduced = tws_all[1:]

    spatio_points_dist = np.delete(spatiotemporal.euclidian_dist_all, 0, 0)
    spatio_points_dist = np.delete(spatio_points_dist, 0, 1)

    spatiotemporal_points_dist = np.delete(spatiotemporal.spatiotemporal_dist_all, 0, 0)
    spatiotemporal_points_dist = np.delete(spatiotemporal_points_dist, 0, 1)

    if distance == 'spatiotemp':
        solver = Solver(Z, spatiotemporal_points_dist, P, ng, Pc, Pm, Pmb, k=k)
    else:
        solver = Solver(Z, spatio_points_dist, P, ng, Pc, Pm, Pmb, k=k)

    result = solver.solve()

    res_dataset = np.array([[test_dataset_points[point] for point in cluster] for cluster in result])
    res_tws = np.array([[tws_reduced[point] for point in cluster] for cluster in result])

    dist, wait_time, late_time = estimate_solution(result, spatiotemporal_points_dist)
    print("Total distance: {}".format(dist))

    if plot:
        plot_clusters(test_dataset_points, res_dataset, res_tws, spatiotemporal.MAX_TW)


def solve_test(k=None):
    # init test data
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

    solve(test_dataset, tws_all, service_time_all, k=k, distance='spatiotemp')


def solve_c101(k=None):
    c101_dataset = pd.read_fwf('data/c101_mod.txt')

    tws_all = np.empty((0, 2))
    service_time_all = np.empty((0, 1))
    points_dataset = np.empty((0, 2))

    for i in range(c101_dataset.shape[0]):
        tws_all = np.concatenate((tws_all, [[c101_dataset['READY_TIME'][i],
                                            c101_dataset['DUE_DATE'][i]]]), axis=0)

        service_time_all = np.concatenate((service_time_all, [[c101_dataset['SERVICE_TIME'][i]]]), axis=0)

        points_dataset = np.concatenate((points_dataset, [[c101_dataset['XCOORD'][i],
                                                           c101_dataset['YCOORD'][i]]]), axis=0)

    solve(points_dataset, tws_all, service_time_all, k=int(c101_dataset['VEHICLE_NUMBER'][0]), distance='spatiotemp', plot=False)


def solve_r101(k=None):
    r101_dataset = pd.read_fwf('data/r101_reduced.txt')

    tws_all = np.empty((0, 2))
    service_time_all = np.empty((0, 1))
    points_dataset = np.empty((0, 2))

    for i in range(r101_dataset.shape[0]):
        tws_all = np.concatenate((tws_all, [[r101_dataset['READY_TIME'][i],
                                             r101_dataset['DUE_DATE'][i]]]), axis=0)

        service_time_all = np.concatenate((service_time_all, [[r101_dataset['SERVICE_TIME'][i]]]), axis=0)

        points_dataset = np.concatenate((points_dataset, [[r101_dataset['XCOORD'][i],
                                                           r101_dataset['YCOORD'][i]]]), axis=0)

    solve(points_dataset, tws_all, service_time_all, k=int(r101_dataset['VEHICLE_NUMBER'][0]), distance='spatiotemp', plot=True)


def plot_clusters(init_dataset, dataset, tws, max_tw):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize_standart,
                             dpi=dpi_standart, subplot_kw={'projection': '3d'})

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

    colors = [rgb2hex([np.random.random_sample(), np.random.random_sample(), np.random.random_sample()])
              for _ in dataset]

    for i in range(dataset.shape[0]):
        plot_with_tws(dataset[i], tws[i], max_tw, colors[i], axes)

    # for i, data in enumerate(init_dataset):
    #     axes.text(data[0], data[1], 0.0, str(i))

    plt.show()


def plot_with_tws(spatial_data, tws, max_tw, colors, axes):
    cluster_size = spatial_data[0].size

    x_data = np.array([i[0] for i in spatial_data])
    y_data = np.array([i[1] for i in spatial_data])

    z_data1 = np.array([i[0] for i in tws])
    z_data2 = np.array([i[1] for i in tws])

    dz_data = np.abs(np.subtract(z_data1, z_data2))

    axes.scatter(x_data, y_data, 0.0, c=colors, s=cluster_size)
    axes.scatter(x_data, y_data, z_data1, c=colors, s=cluster_size)
    axes.scatter(x_data, y_data, z_data2, c=colors, s=cluster_size)

    axes.bar3d(x_data - depth / 8., y_data - depth / 8., 0.0, width / 4., depth / 4., max_tw)
    axes.bar3d(x_data - depth / 2., y_data - depth / 2., z_data1, width, depth, dz_data)

    axes.plot(x_data, y_data, 0.0, linewidth=linewidth_standart)

    axes.set_zlim(0, None)





if __name__ == '__main__':
    # solve_test(k=2)

    # solve_c101()

    solve_r101()
