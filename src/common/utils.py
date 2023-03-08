import argparse
import pathlib
from time import time

import numpy as np
from mpl_toolkits.mplot3d.axis3d import Axis
from sklearn.metrics.pairwise import euclidean_distances

# fix wrong z-offsets in 3d plot
from src.common.dataset import Dataset


# TODO: испытать, обязательно ли должно быть в main'e
def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0, 0, 1.0 / 4]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs


def set_options():
    if not hasattr(Axis, "_get_coord_info_old"):
        Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--launch_entries', action='store', type=str,
                            help='Input file with launch entries to solve')
    arg_parser.add_argument('--solve_cluster', action='store', type=str,
                            help='Launch cluster in one of modes: default, sequential, dm')
    arg_parser.add_argument('--solve_tsptw', action='store', type=str,
                            help='Launch tsptw in one of modes: default, sequential, dm')
    arg_parser.add_argument('--plot_stats', action='store_true',
                            help='Turn on plot_stats mode, stats array to plot is in configuration')
    arg_parser.add_argument('--plot_solutions', action='store_true',
                            help='Turn on plot_solutions mode')

    arguments = arg_parser.parse_args()

    return arguments


def calc_euclidian_dist(points_all: np.ndarray, i, j):
    return np.linalg.norm(points_all[i] - points_all[j])


def calc_euclidian_dist_all(points_all: np.ndarray):
    return euclidean_distances(points_all, points_all)


def timing(output):
    def decorator(f):
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()

            with open(output, 'w') as file:
                file.write('{}\n'.format(round(te - ts, 4)))
            return result

        return wrap

    return decorator


def create_directory(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def read_standard_dataset(dataset):
    points_dataset = np.empty((0, 2))
    tws_all = np.empty((0, 2))
    service_time_all = np.empty((0, 1))

    for i in range(dataset.shape[0]):
        tws_all = np.concatenate((tws_all, [[dataset['READY_TIME'][i],
                                             dataset['DUE_DATE'][i]]]), axis=0)

        service_time_all = np.concatenate((service_time_all, [[dataset['SERVICE_TIME'][i]]]), axis=0)

        points_dataset = np.concatenate((points_dataset, [[dataset['XCOORD'][i],
                                                           dataset['YCOORD'][i]]]), axis=0)

    return points_dataset, tws_all, service_time_all


def get_common_entry_id(k3, dataset: Dataset):
    if dataset is not None:
        return '_'.join([dataset.name, str(dataset.dim),
                         dataset.dataset_type, str(int(k3))])
    return None


# Прочитать маршрут bks
def read_standard_route(full_path):
    routes = []

    with open(full_path) as f:
        for line in f:
            route = [int(i) for i in line.split()]
            routes.append(route)

    return routes
