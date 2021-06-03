import numpy as np
import pandas as pd
import sys


class Statistics:
    def __init__(self, testing_datasets, dims_array, k3_array):
        self._testing_datasets = testing_datasets
        self._dims_array = dims_array
        self._k3_array = k3_array

        self._BASE_DIR = sys.path[0]

    def collect_time_data(self):
        different_k3_arr = []
        for k3 in self._k3_array:
            dataset_series_array = []
            for dataset_series in self._testing_datasets:
                dataset_time_data = []
                for dataset in dataset_series:
                    time_cluster = pd.read_fwf(
                        self._BASE_DIR + '/result/cluster_result/' + dataset['name'] + '_output_{}/time_cluster.csv'.format(int(k3)),
                        header=None)
                    time_tsp = pd.read_fwf(
                        self._BASE_DIR + '/result/tsptw_result/' + dataset['name'] + '_output_{}/time_tsp.csv'.format(int(k3)), header=None)
                    time_common = time_cluster.values[0][0] + time_tsp.values[0][0]

                    dataset_time_data.append(time_common)
                dataset_series_array.append(dataset_time_data)
            different_k3_arr.append(dataset_series_array)

        return different_k3_arr

    def collect_evaluation(self):
        different_k3_arr_dist = []
        different_k3_arr_wait_time = []
        different_k3_arr_late_time = []
        different_k3_arr_eval = []
        for k3 in self._k3_array:
            dataset_series_array_dist = []
            dataset_series_array_wait_time = []
            dataset_series_array_late_time = []
            dataset_series_array_eval = []
            for dataset_series in self._testing_datasets:
                dataset_data_dist = []
                dataset_data_wait_time = []
                dataset_data_late_time = []
                dataset_data_eval = []
                for dataset in dataset_series:
                    evaluation = pd.read_fwf(
                        self._BASE_DIR + '/result/evaluation/' + dataset['name'] + '_output_{}/evaluation.csv'.format(int(k3)), header=None).values

                    evaluation = [row[0] for row in evaluation]

                    dataset_data_dist.append(evaluation[0])
                    dataset_data_wait_time.append(evaluation[1])
                    dataset_data_late_time.append(evaluation[2])
                    dataset_data_eval.append(evaluation[3])
                dataset_series_array_dist.append(dataset_data_dist)
                dataset_series_array_wait_time.append(dataset_data_wait_time)
                dataset_series_array_late_time.append(dataset_data_late_time)
                dataset_series_array_eval.append(dataset_data_eval)
            different_k3_arr_dist.append(dataset_series_array_dist)
            different_k3_arr_wait_time.append(dataset_series_array_wait_time)
            different_k3_arr_late_time.append(dataset_series_array_late_time)
            different_k3_arr_eval.append(dataset_series_array_eval)

        return different_k3_arr_dist, different_k3_arr_wait_time, different_k3_arr_late_time, different_k3_arr_eval

    def collect_time_stats(self):
        different_k3_arr_wait_time = []
        different_k3_arr_late_time = []

        for k3 in self._k3_array:
            dataset_series_array_wait_time = []
            dataset_series_array_late_time = []
            for dataset_series in self._testing_datasets:
                dataset_data_wait_time = []
                dataset_data_late_time = []
                for i, dataset in enumerate(dataset_series):
                    evaluation = pd.read_fwf(
                        self._BASE_DIR + '/result/evaluation/' + dataset['name'] + '_output_{}/evaluation.csv'.format(int(k3)), header=None).values

                    evaluation = [row[0] for row in evaluation]

                    dataset_data_wait_time.append(evaluation[1] / self._dims_array[i])
                    dataset_data_late_time.append(evaluation[2] / self._dims_array[i])
                dataset_series_array_wait_time.append(dataset_data_wait_time)
                dataset_series_array_late_time.append(dataset_data_late_time)
            different_k3_arr_wait_time.append(dataset_series_array_wait_time)
            different_k3_arr_late_time.append(dataset_series_array_late_time)

        return different_k3_arr_wait_time, different_k3_arr_late_time

    def collect_additional_times(self):
        different_k3_arr_max_wait_time = []
        different_k3_arr_max_late_time = []
        different_k3_arr_wait_time_part = []
        different_k3_arr_late_time_part = []

        for k3 in self._k3_array:
            dataset_series_array_max_wait_time = []
            dataset_series_array_max_late_time = []
            dataset_series_array_wait_time_part = []
            dataset_series_array_late_time_part = []

            for dataset_series in self._testing_datasets:
                dataset_data_max_wait_time = []
                dataset_data_max_late_time = []
                dataset_data_wait_time_part = []
                dataset_data_late_time_part = []

                for i, dataset in enumerate(dataset_series):
                    data = pd.read_fwf(self._BASE_DIR + '/data/' + dataset['data_file'])
                    vehicle_number = int(data['VEHICLE_NUMBER'][0])

                    max_wait_time = 0.0
                    max_late_time = 0.0

                    wait_counter = 0
                    late_counter = 0

                    for j in range(vehicle_number):
                        cur_report = pd.read_csv(self._BASE_DIR + '/result/tsptw_result/' + dataset['name'] + '_output_{}/report{}.csv'.format(int(k3), j), delimiter=' ')

                        wait_late_df = cur_report[['Wait_Time', 'Late_Time']].values
                        for row in wait_late_df[:-2]:
                            if float(row[0]) > max_wait_time:
                                max_wait_time = float(row[0])
                            if float(row[1]) > max_late_time:
                                max_late_time = float(row[1])

                            if float(row[0]) > 0.0:
                                wait_counter += 1
                            if float(row[1]) > 0.0:
                                late_counter += 1

                    dataset_data_max_wait_time.append(max_wait_time)
                    dataset_data_max_late_time.append(max_late_time)
                    dataset_data_wait_time_part.append(wait_counter / self._dims_array[i])
                    dataset_data_late_time_part.append(late_counter / self._dims_array[i])
                dataset_series_array_max_wait_time.append(dataset_data_max_wait_time)
                dataset_series_array_max_late_time.append(dataset_data_max_late_time)
                dataset_series_array_wait_time_part.append(dataset_data_wait_time_part)
                dataset_series_array_late_time_part.append(dataset_data_late_time_part)
            different_k3_arr_max_wait_time.append(dataset_series_array_max_wait_time)
            different_k3_arr_max_late_time.append(dataset_series_array_max_late_time)
            different_k3_arr_wait_time_part.append(dataset_series_array_wait_time_part)
            different_k3_arr_late_time_part.append(dataset_series_array_late_time_part)

        return different_k3_arr_max_wait_time, different_k3_arr_max_late_time, different_k3_arr_wait_time_part, different_k3_arr_late_time_part
