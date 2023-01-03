import numpy as np
import pandas as pd

from src.common.utils import read_standard_dataset, calc_euclidian_dist_all, read_standard_route
from src.vrptw.vrptw_launch_entry import VRPTWLaunchEntry


class Statistics:
    def __init__(self, vrptw_launch_entry: VRPTWLaunchEntry):
        self._vrptw_launch_entry = vrptw_launch_entry
        self._cluster_launch_entry_arr = self._vrptw_launch_entry.cluster_launch_entry_arr
        self._tsptw_launch_entry_arr = self._vrptw_launch_entry.tsptw_launch_entry_arr

    def collect_all_stats(self):
        common_id_arr = []
        dim_arr = []
        k3_arr = []
        dataset_type_arr = []
        dataset_name_arr = []

        time_cluster_arr = []
        time_tsptw_arr = []
        time_common_arr = []

        evaluation_arr = np.empty(shape=(self._tsptw_launch_entry_arr.size, 4))
        wait_time_arr = []
        late_time_arr = []

        max_wait_time_arr = []
        max_late_time_arr = []
        wait_time_part_arr = []
        late_time_part_arr = []

        total_time_arr = []

        std_wait_arr = []

        # tsptw_launch_entry и cluster_launch_entry соответствуют друг другу
        for i, tsptw_launch_entry in enumerate(self._tsptw_launch_entry_arr):
            # TODO: только ради vehicle_number, может поместить куда-то количество тс?
            data = pd.read_fwf(self._vrptw_launch_entry.BASE_DIR + '/input/task/'
                               + tsptw_launch_entry.dataset.data_file)
            vehicle_number = int(data['VEHICLE_NUMBER'][0])

            self._fill_time_stats(tsptw_launch_entry, time_cluster_arr, time_tsptw_arr, time_common_arr)
            self._fill_evaluation_data(tsptw_launch_entry, vehicle_number, i,
                                       evaluation_arr, wait_time_arr, late_time_arr)
            self._fill_additional_time_stats(tsptw_launch_entry, vehicle_number, max_wait_time_arr, max_late_time_arr,
                                             wait_time_part_arr, late_time_part_arr, total_time_arr, std_wait_arr)

            common_id_arr.append(tsptw_launch_entry.common_id)
            dim_arr.append(tsptw_launch_entry.dataset.dim)
            k3_arr.append(tsptw_launch_entry.cluster_k3)
            dataset_type_arr.append(tsptw_launch_entry.dataset.dataset_type)
            dataset_name_arr.append(tsptw_launch_entry.dataset.name)

        return pd.DataFrame({
            'common_id': common_id_arr,
            'name': dataset_name_arr,
            'dim': dim_arr,
            'k3': k3_arr,
            'dataset_type': dataset_type_arr,
            'time_cluster': time_cluster_arr,
            'time_tsptw': time_tsptw_arr,
            'time_common': time_common_arr,
            'distance': evaluation_arr[:, 0],
            'wait_time': evaluation_arr[:, 1],
            'late_time': evaluation_arr[:, 2],
            'eval': evaluation_arr[:, 3],
            'wait_time_per_vehicle': wait_time_arr,
            'late_time_per_customer': late_time_arr,
            'max_wait_time': max_late_time_arr,
            'wait_time_part': wait_time_part_arr,
            'late_time_part': late_time_part_arr,
            'total_time': total_time_arr,
        })

    def _fill_time_stats(self, launch_entry, time_cluster_arr, time_tsptw_arr, time_common_arr):
        time_cluster = pd.read_fwf(
            self._vrptw_launch_entry.CLUSTER_OUTPUT + launch_entry.common_id + '/time_cluster.csv',
            header=None
        )
        time_tsptw = pd.read_fwf(
            self._vrptw_launch_entry.TSPTW_OUTPUT + launch_entry.common_id + '/time_tsptw.csv',
            header=None
        )
        time_cluster_arr.append(time_cluster.values[0][0])
        time_tsptw_arr.append(time_tsptw.values[0][0])
        time_common_arr.append(time_cluster.values[0][0] + time_tsptw.values[0][0])

    def _fill_evaluation_data(self, launch_entry, vehicle_number, i, evaluation_arr, wait_time_arr, late_time_arr):
        evaluation_data = pd.read_fwf(
            self._vrptw_launch_entry.EVALUATION_OUTPUT + launch_entry.common_id + '/evaluation.csv',
            header=None
        )
        evaluation_arr[i] = evaluation_data.transpose().values[0]

        wait_time_arr.append(evaluation_arr[i][1] / vehicle_number)
        late_time_arr.append(evaluation_arr[i][2] / launch_entry.dataset.dim)

    def _fill_additional_time_stats(self, launch_entry, vehicle_number,
                                    max_wait_time_arr, max_late_time_arr, wait_time_part_arr,
                                    late_time_part_arr, total_time_arr, std_wait_arr):
        max_wait_time = 0.0
        max_late_time = 0.0

        wait_counter = 0
        late_counter = 0

        total_travel_time = 0.0

        wait_arr_route = []
        for j in range(vehicle_number):
            cur_report = pd.read_csv(self._vrptw_launch_entry.TSPTW_OUTPUT
                                     + launch_entry.common_id
                                     + '/report{}.csv'.format(j), delimiter=' ')

            time_df = cur_report[['Wait_Time', 'Late_Time', 'Leave_Time']].values
            for row in time_df[:-2]:
                if float(row[0]) > max_wait_time:
                    max_wait_time = float(row[0])
                if float(row[1]) > max_late_time:
                    max_late_time = float(row[1])

                if float(row[0]) > 0.0:
                    wait_counter += 1
                if float(row[1]) > 0.0:
                    late_counter += 1
            total_travel_time += float(time_df[-1][2])
            wait_arr_route.append(float(time_df[-1][0]))

        max_wait_time_arr.append(max_wait_time)
        max_late_time_arr.append(max_late_time)
        wait_time_part_arr.append(wait_counter / launch_entry.dataset.dim)
        late_time_part_arr.append(late_counter / launch_entry.dataset.dim)

        total_time_arr.append(total_travel_time / vehicle_number)

        std_wait_arr.append(np.std(wait_arr_route))

    ### collect stats from bks
    def collect_bks_stats(self):
        bks_routes_df = self._parse_bks_routes()

        wait_time_sum_arr = []
        late_time_sum_arr = []
        total_time_sum_arr = []
        distance_sum_arr = []

        add_wait_arr = []
        add_late_arr = []
        add_total_arr = []

        std_wait_arr_bns = []

        dataset_name_arr = []

        grouped = bks_routes_df.groupby('dataset')

        for name, group in grouped:
            ### evaluation_stats
            wait_time_sum = group['wait_time'].sum()
            late_time_sum = group['late_time'].sum()
            total_time_sum = group['total_time'].sum()
            distance_sum = group['distance'].sum()

            wait_time_sum_arr.append(wait_time_sum)
            late_time_sum_arr.append(late_time_sum)
            total_time_sum_arr.append(total_time_sum)
            distance_sum_arr.append(distance_sum)

            ### additional time_stats
            vehicle_number = group.shape[0]
            dim = group['dim'].iloc[0]

            add_wait_arr.append(wait_time_sum / vehicle_number)
            add_late_arr.append(late_time_sum / dim)
            add_total_arr.append(total_time_sum / vehicle_number)

            ### std
            std_wait_arr_bns.append(np.std(group['wait_time'].values))

            dataset_name_arr.append(name)

        return pd.DataFrame({
            'name': dataset_name_arr,
            'wait_time_bks': wait_time_sum_arr,
            'late_time_bks': late_time_sum_arr,
            'total_time_bks': total_time_sum_arr,
            'distance_bks': distance_sum_arr,
        })

    ### parse bks and save stats
    def _parse_bks_routes(self):
        dataset_arr = self._vrptw_launch_entry.dataset_arr

        wait_time_all = np.array([])
        late_time_all = np.array([])
        total_time_all = np.array([])
        dist_all = np.array([])

        dataset_name_all = np.array([])
        dim_all = np.array([])

        for d in dataset_arr:
            dataset = pd.read_fwf(self._vrptw_launch_entry.BASE_DIR + '/input/task/'
                                  + d.data_file)

            points_dataset, tws_all, service_time_all = read_standard_dataset(dataset)
            euclidian_dist = calc_euclidian_dist_all(points_dataset)

            routes_dataset = read_standard_route(self._vrptw_launch_entry.BASE_DIR + '/input/bks/' + d.data_file)
            routes_len = len(routes_dataset)
            wait_time_arr = np.zeros(routes_len)
            late_time_arr = np.zeros(routes_len)
            total_time_arr = np.zeros(routes_len)
            dist_arr = np.zeros(routes_len)

            dataset_name_arr = np.full(routes_len, d.name)
            dim_arr = np.full(routes_len, d.dim)

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

                wait_time_arr[i] = cur_wait_time
                late_time_arr[i] = cur_late_time
                total_time_arr[i] = cur_time

                dist_arr[i] = cur_distance
                dataset_name_arr[i] = d.name
                dim_arr[i] = d.dim
            wait_time_all = np.append(wait_time_all, wait_time_arr)
            late_time_all = np.append(late_time_all, late_time_arr)
            total_time_all = np.append(total_time_all, total_time_arr)
            dist_all = np.append(dist_all, dist_arr)

            dataset_name_all = np.append(dataset_name_all, dataset_name_arr)
            dim_all = np.append(dim_all, dim_arr)

        return pd.DataFrame({
            'dataset': dataset_name_all,
            'dim': dim_all,
            'wait_time': wait_time_all,
            'late_time': late_time_all,
            'total_time': total_time_all,
            'distance': dist_all
        })
