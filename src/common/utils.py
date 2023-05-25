import argparse
import pathlib
from time import time

import numpy as np
from mpl_toolkits.mplot3d.axis3d import Axis
from numpy import inf
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


def make_cluster_from_medoids(distances, dm_priority_list, medoids):
    points_size = int(np.ceil(distances[0].size / medoids.size))
    result = np.full((medoids.size, points_size), -1)
    costs_sum = 0.0

    approved = np.arange(distances[0].size)

    # Убрать медоиды из поиска, вручную их помещаем в кластер на нулевую позицию
    approved = np.delete(approved, np.ravel([np.where(approved == med) for med in medoids]))

    # Сформировать список из паттернов
    all_priority_list = np.array([list(item[1]) for item in dm_priority_list], dtype=list)

    # Если есть паттерны для распределения
    if all_priority_list.size != 0:
        # Для каждого кластера
        for i, gene in enumerate(medoids):
            # Сначала заполнить медоиды
            result[i][0] = gene
            # Поиск паттерна, суммарное расстояние от каждой вершины которого до текущего медоида минимально
            # Медоид может быть в паттерне - учитывается при занесении вершин в кластеры
            # Если медоид в паттерне, то расстояние будет нулевым, что повысит вероятность попадания вершин из паттерна в его кластер
            cur_near_pattern_ind = -1
            cur_min_priority_sum = inf
            for ind, pattern in enumerate(all_priority_list):
                priority_sums = np.sum([distances[gene][ii] for ii in pattern])
                if priority_sums < cur_min_priority_sum:
                    cur_min_priority_sum = priority_sums
                    cur_near_pattern_ind = ind

            # Если паттерн по длине вдруг оказался больше размера кластера (по идее практически невозможно)
            best_pattern = all_priority_list[cur_near_pattern_ind]
            if len(best_pattern) > points_size:
                all_priority_list[cur_near_pattern_ind] = best_pattern[cur_near_pattern_ind:points_size]

            j = 1
            for el in best_pattern:
                # Размещаем все вершины, если они входят в список разрешенных
                if el in approved:
                    result[i][j] = el
                    costs_sum += distances[gene][el]
                    j += 1

                    # Удалить размещенную вершину из списка разрешенных
                    approved = approved[approved != el]

    for i, gene in enumerate(medoids):
        # Сразу помещаем медоид
        result[i][0] = gene
        # Идем только по незаполненным
        j_inds = [ind for ind, e in enumerate(result[i]) if e == -1]
        for j in j_inds:
            if approved.size != 0:
                # Строка с расстояниями до других вершин
                cur_dist = distances[gene]

                # Ищем индекс ближайшей вершины
                cur_min_ind = -1
                cur_min = np.inf
                for ind, el in enumerate(cur_dist):
                    if ind in approved and el < cur_min:
                        cur_min = el
                        cur_min_ind = ind

                # Не найдем минимум, если он не в approved
                if cur_min != np.inf:
                    costs_sum += cur_min

                    result[i][j] = cur_min_ind

                    # Удаляем из списка разрешенных
                    approved = approved[approved != cur_min_ind]

    return result, costs_sum
