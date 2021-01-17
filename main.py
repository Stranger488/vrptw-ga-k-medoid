import random

import matplotlib.pyplot as plt
import numpy as np
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
Z = 4

# Population size, i. g. number of chromosomes in population
P = 15

# Number of all generations
ng = 15

# Crossover probability
Pc = 0.9

# Mutation probability
Pm = 0.1

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


def solve_test():
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

    # init and calculate all spatiotemporal distances
    spatiotemporal = Spatiotemporal(test_dataset, tws_all, service_time_all, k1, k2, k3, alpha1, alpha2)
    spatiotemporal_dist_all = spatiotemporal.calculate_all_distances()

    spatiotemporal_points_dist = np.delete(spatiotemporal_dist_all, 0, 0)
    spatiotemporal_points_dist = np.delete(spatiotemporal_points_dist, 0, 1)

    test_dataset_points = test_dataset[1:][:]
    tws_reduced = tws_all[1:]

    # print(spatiotemporal_points_dist)
    # print(test_dataset_points)
    # plot_with_tws(test_dataset, spatiotemporal.tws_all, spatiotemporal.MAX_TW)

    solver = Solver(Z, spatiotemporal_points_dist, P, ng, Pc, Pm)
    result = solver.solve()

    res_dataset = np.array([[test_dataset_points[point] for point in cluster] for cluster in result])
    res_tws = np.array([[tws_reduced[point] for point in cluster] for cluster in result])

    plot_clusters(test_dataset_points, res_dataset, res_tws, spatiotemporal.MAX_TW)


def plot_clusters(init_dataset, dataset, tws, max_tw):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize_standart,
                             dpi=dpi_standart, subplot_kw={'projection': '3d'})

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

    colors = [rgb2hex([np.random.random_sample(), np.random.random_sample(), np.random.random_sample()])
              for _ in dataset]

    for i in range(dataset[0][0].size):
        plot_with_tws(dataset[i], tws[i], max_tw, colors[i], axes)

    for i, data in enumerate(init_dataset):
        axes.text(data[0], data[1], 0.0, str(i))

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
    solve_test()
