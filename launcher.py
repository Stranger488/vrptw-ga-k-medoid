from vrptw_solver import VRPTWSolver
from statistics import Statistics
from plot import Plot


class Launcher:
    def __init__(self, in_dataset_series=None, is_solve=False, plot_stats=None, mode=None):
        self._in_dataset_series = __import__(in_dataset_series)
        self._mapping = self._in_dataset_series.mapping
        self._testing_datasets = self._in_dataset_series.testing_datasets
        self._k3_array = self._in_dataset_series.k3_array
        self._dims_array = self._in_dataset_series.dims_array

        self._is_solve = is_solve
        self._plot_stats = plot_stats
        self._mode = mode

        self._plotter = Plot()

    def launch(self):
        if self._is_solve:
            self._make_solving()

        if self._plot_stats:
            self._make_plot_stats()

    def _make_solving(self):
        for dataset_series in self._testing_datasets:
            for dataset in dataset_series:
                base_name = dataset['output_dir'][:len(dataset['output_dir']) - 1]
                for k3 in self._k3_array:
                    vrptw_solver = VRPTWSolver(k3, self._mode)
                    dataset['output_dir'] = base_name + '_' + str(int(k3)) + '/'
                    vrptw_solver.solve_and_plot([dataset, ])

    def _make_plot_stats(self):
        statistics = Statistics(self._testing_datasets, self._dims_array, self._k3_array)

        different_k3_arr_dist, different_k3_arr_wait_time, different_k3_arr_late_time, different_k3_arr_total_time, different_k3_arr_eval = statistics.collect_evaluation()

        different_k3_arr_wait_time_per_customer, different_k3_arr_late_time_per_customer = statistics.collect_time_stats()

        different_k3_arr_max_wait_time, different_k3_arr_max_late_time, different_k3_arr_wait_time_part, different_k3_arr_late_time_part = statistics.collect_additional_times()

        wait_arr, late_arr, total_arr, dist_arr = statistics.collect_bns_data()
        avg_wait_arr, avg_late_arr = statistics.collect_bns_additional_data()

        if self._plot_stats == 'time':
            different_k3_arr = statistics.collect_time_data()
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array, different_k3_arr,
                                    xlabel='Число клиентов',
                                    ylabel='Время выполнения программы, с', mapping=self._mapping)
        elif self._plot_stats == 'distance':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array, different_k3_arr_dist,
                                    xlabel='Число клиентов',
                                    ylabel='Пройденное расстояние', mapping=self._mapping)
        elif self._plot_stats == 'wait_time':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array,
                                    different_k3_arr_wait_time, xlabel='Число клиентов',
                                    ylabel='Время ожидания', mapping=self._mapping)
        elif self._plot_stats == 'late_time':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array,
                                    different_k3_arr_late_time, xlabel='Число клиентов',
                                    ylabel='Время опоздания', mapping=self._mapping)
        elif self._plot_stats == 'total_evaluation':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array, different_k3_arr_eval,
                                    xlabel='Число клиентов',
                                    ylabel='Общая оценка', mapping=self._mapping)
        elif self._plot_stats == 'avg_wait_time':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array,
                                    different_k3_arr_wait_time_per_customer,
                                    xlabel='Число клиентов',
                                    ylabel='Время ожидания на одно ТС', mapping=self._mapping)
        elif self._plot_stats == 'avg_late_time':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array,
                                    different_k3_arr_late_time_per_customer,
                                    xlabel='Число клиентов',
                                    ylabel='Время опоздания на одного клиента', mapping=self._mapping)
        elif self._plot_stats == 'max_wait_time':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array,
                                    different_k3_arr_max_wait_time,
                                    xlabel='Число клиентов',
                                    ylabel='Максимальное время ожидания', mapping=self._mapping)
        elif self._plot_stats == 'max_late_time':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array,
                                    different_k3_arr_max_late_time,
                                    xlabel='Число клиентов',
                                    ylabel='Максимальное время опоздания', mapping=self._mapping)
        elif self._plot_stats == 'wait_time_part':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array,
                                    different_k3_arr_wait_time_part,
                                    xlabel='Число клиентов',
                                    ylabel='Доля ожиданий среди всех вершин', mapping=self._mapping)
        elif self._plot_stats == 'late_time_part':
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array,
                                    different_k3_arr_late_time_part,
                                    xlabel='Число клиентов',
                                    ylabel='Доля опозданий среди всех вершин', mapping=self._mapping)
        elif self._plot_stats == 'distance_bns':
            self._plotter.plot_dataset_series_with_bns(self._testing_datasets, self._dims_array,
                                                       different_k3_arr_dist[0], dist_arr,
                                                       xlabel='Число клиентов',
                                                       ylabel='Пройденное расстояние (сравнение с наилучшими решениями)', mapping=self._mapping)
        elif self._plot_stats == 'wait_time_bns':
            self._plotter.plot_dataset_series_with_bns(self._testing_datasets, self._dims_array,
                                                       different_k3_arr_wait_time[0], wait_arr,
                                                       xlabel='Число клиентов',
                                                       ylabel='Время ожидания (сравнение с наилучшими решениями)', mapping=self._mapping)
        elif self._plot_stats == 'late_time_bns':
            self._plotter.plot_dataset_series_with_bns(self._testing_datasets, self._dims_array,
                                                       different_k3_arr_late_time[0], late_arr,
                                                       xlabel='Число клиентов',
                                                       ylabel='Время опоздания (сравнение с наилучшими решенияим)', mapping=self._mapping)
        elif self._plot_stats == 'total_time_bns':
            self._plotter.plot_dataset_series_with_bns(self._testing_datasets, self._dims_array,
                                                       different_k3_arr_total_time[0], total_arr,
                                                       xlabel='Число клиентов',
                                                       ylabel='Время в пути (сравнение с наилучшими решенияим)', mapping=self._mapping)                                                       
        elif self._plot_stats == 'avg_wait_time_bns':
            self._plotter.plot_dataset_series_with_bns(self._testing_datasets, self._dims_array,
                                                       different_k3_arr_wait_time_per_customer[0], avg_wait_arr,
                                                       xlabel='Число клиентов',
                                                       ylabel='Время ожидания на одно ТС (сравнение с наилучшими решениями)', mapping=self._mapping)
        elif self._plot_stats == 'avg_late_time_bns':
            self._plotter.plot_dataset_series_with_bns(self._testing_datasets, self._dims_array,
                                                       different_k3_arr_late_time_per_customer[0], avg_late_arr,
                                                       xlabel='Число клиентов',
                                                       ylabel='Время опоздания на одного клиента (сравнение с наилучшими решенияим)', mapping=self._mapping)
        else:
            print('Unrecognized plot_stats parameter. Setting it to distance...')
            self._plotter.plot_data(self._testing_datasets, self._dims_array, self._k3_array, different_k3_arr_dist,
                                    xlabel='Число клиентов',
                                    ylabel='Пройденное расстояние, с', mapping=self._mapping)

    def _make_bns_plots(self):
        statistics = Statistics(self._testing_datasets, self._dims_array, self._k3_array)
        wait_arr, late_arr, dist_arr = statistics.collect_bns_data()
        # self._plotter.plot_data_bns(self._testing_datasets, self._dims_array, wait_arr)
        self._plotter.plot_data_bns(self._testing_datasets, self._dims_array, dist_arr)
        # self._plotter.plot_data_bns(self._testing_datasets, self._dims_array, late_arr)
