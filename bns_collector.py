import sys
import math
import pathlib

import numpy as np
import pandas as pd


class BNSCollector:
    def __init__(self):
        self._BASE_DIR = sys.path[0]

    def _euclidian_distance(self, points_all, i, j):
        if i != j:
            sum_all = 0
            for k in range(len(points_all[i])):
                square = pow(points_all[j][k] - points_all[i][k], 2)
                sum_all += square

            sqr = math.sqrt(sum_all)
            return sqr

        return 0.0

    def _calculate_euclidian_dist_all(self, points_all):
        length = len(points_all)
        euclidian_dist_all = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                if j >= i:
                    euclidian_dist_all[i, j] = self._euclidian_distance(points_all, i, j)
                else:
                    euclidian_dist_all[i, j] = euclidian_dist_all[j, i]

        return euclidian_dist_all

    def _read_standard_dataset(self, dataset, points_dataset, tws_all, service_time_all):
        for i in range(dataset.shape[0]):
            tws_all = np.concatenate((tws_all, [[dataset['READY_TIME'][i],
                                                 dataset['DUE_DATE'][i]]]), axis=0)

            service_time_all = np.concatenate((service_time_all, [[dataset['SERVICE_TIME'][i]]]), axis=0)

            points_dataset = np.concatenate((points_dataset, [[dataset['XCOORD'][i],
                                                               dataset['YCOORD'][i]]]), axis=0)

        return points_dataset, tws_all, service_time_all

    def _read_standard_route(self, filename):
        routes = []

        with open(self._BASE_DIR + '/routes/' + filename) as f:
            for line in f:
                route = [int(i) for i in line.split()]
                routes.append(route)

        return routes

    def parse_results(self, filename):
        output_dir = filename[:-4] + '/'
        pathlib.Path(self._BASE_DIR + '/bns_wait_time/' + output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self._BASE_DIR + '/bns_late_time/' + output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self._BASE_DIR + '/bns_total_time/' + output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self._BASE_DIR + '/bns_dist/' + output_dir).mkdir(parents=True, exist_ok=True)

        dataset = pd.read_fwf(self._BASE_DIR + '/data/' + filename)

        points_dataset = np.empty((0, 2))
        tws_all = np.empty((0, 2))
        service_time_all = np.empty((0, 1))
        points_dataset, tws_all, service_time_all = self._read_standard_dataset(dataset, points_dataset, tws_all,
                                                                                service_time_all)
        euclidian_dist = self._calculate_euclidian_dist_all(points_dataset)

        routes_dataset = self._read_standard_route(filename)
        wait_time_dataset = np.zeros(len(routes_dataset))
        late_time_dataset = np.zeros(len(routes_dataset))
        total_time_dataset = np.zeros(len(routes_dataset))
        dist_dataset = np.zeros(len(routes_dataset))

        for i, route in enumerate(routes_dataset):
            cur_time = 0.0
            cur_wait_time = 0.0
            cur_late_time = 0.0
            prev = 0
            cur_distance = 0.0
            for j in route:
                cur_distance += euclidian_dist[prev][j]

                cur_time += euclidian_dist[prev][j]
                wait_time = tws_all[j][0] - cur_time
                late_time = cur_time - tws_all[j][1]
                if wait_time > 0.0:
                    cur_wait_time += wait_time
                    cur_time = tws_all[j][0]
                if late_time > 0.0:
                    cur_late_time += late_time

                cur_time += service_time_all[j]
                prev = j
            cur_time += euclidian_dist[prev][0]
            cur_distance += euclidian_dist[prev][0]

            wait_time_dataset[i] = cur_wait_time
            late_time_dataset[i] = cur_late_time
            total_time_dataset[i] = cur_time

            dist_dataset[i] = cur_distance

        wait_time_df = pd.DataFrame(wait_time_dataset)
        late_time_df = pd.DataFrame(late_time_dataset)
        total_time_df = pd.DataFrame(total_time_dataset)
        dist_df = pd.DataFrame(dist_dataset)
        
        dist_df[1] = 0
        dist_df.loc[0, 1] = sum(dist_dataset)

        wait_time_df.to_csv(self._BASE_DIR + '/bns_wait_time/' + output_dir + 'wait_times.txt', index=False,
                            sep=' ')
        late_time_df.to_csv(self._BASE_DIR + '/bns_late_time/' + output_dir + 'late_times.txt', index=False,
                            sep=' ')
        total_time_df.to_csv(self._BASE_DIR + '/bns_total_time/' + output_dir + 'total_times.txt', index=False,
                            sep=' ')                    
        dist_df.to_csv(self._BASE_DIR + '/bns_dist/' + output_dir + 'distances.txt', index=False,
                       sep=' ')


if __name__ == '__main__':
    bns_collector = BNSCollector()

    bns_collector.parse_results('c104_mod.txt')
    bns_collector.parse_results('C1_2_4_mod.txt')
    bns_collector.parse_results('C1_4_4_mod.txt')
    bns_collector.parse_results('C1_6_4_mod.txt')
    bns_collector.parse_results('C1_8_4_mod.txt')
    bns_collector.parse_results('C1_10_4_mod.txt')

    bns_collector.parse_results('r110_mod.txt')
    bns_collector.parse_results('R1_2_10_mod.txt')
    bns_collector.parse_results('R1_4_10_mod.txt')
    bns_collector.parse_results('R1_6_10_mod.txt')
    bns_collector.parse_results('R1_8_10_mod.txt')
    bns_collector.parse_results('R1_10_10_mod.txt')

    bns_collector.parse_results('rc103_mod.txt')
    bns_collector.parse_results('RC1_2_3_mod.txt')
    bns_collector.parse_results('RC1_4_3_mod.txt')
    bns_collector.parse_results('RC1_6_3_mod.txt')
    bns_collector.parse_results('RC1_8_3_mod.txt')
    bns_collector.parse_results('RC1_10_3_mod.txt')
