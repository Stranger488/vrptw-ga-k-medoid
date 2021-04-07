import pandas as pd

from plot import Plot


class Statistics:
    def __init__(self, testing_datasets, dims_array, k3_array):
        self.plotter = Plot()

        self.testing_datasets = testing_datasets
        self.dims_array = dims_array
        self.k3_array = k3_array

    def collect_time_data(self):
        different_k3_arr = []
        for k3 in self.k3_array:
            dataset_series_array = []
            for dataset_series in self.testing_datasets:
                dataset_time_data = []
                for dataset in dataset_series:
                    time_cluster = pd.read_fwf(
                        'cluster_result/' + dataset['name'] + '_output_{}/time_cluster.csv'.format(int(k3)),
                        header=None)
                    time_tsp = pd.read_fwf(
                        'tsptw_result/' + dataset['name'] + '_output_{}/time_tsp.csv'.format(int(k3)), header=None)
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
        for k3 in self.k3_array:
            dataset_series_array_dist = []
            dataset_series_array_wait_time = []
            dataset_series_array_late_time = []
            dataset_series_array_eval = []
            for dataset_series in self.testing_datasets:
                dataset_data_dist = []
                dataset_data_wait_time = []
                dataset_data_late_time = []
                dataset_data_eval = []
                for dataset in dataset_series:
                    evaluation = pd.read_fwf(
                        'evaluation/' + dataset['name'] + '_output_{}/evaluation.csv'.format(int(k3)), header=None).values

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
