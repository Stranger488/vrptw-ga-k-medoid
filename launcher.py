from kernel import Kernel
from statistics import Statistics
from plot import Plot


class Launcher:
    def __init__(self, in_dataset_series=None, is_solve=False, plot_stats=True):
        self.in_dataset_series = __import__(in_dataset_series)
        self.testing_datasets = self.in_dataset_series.testing_datasets
        self.k3_array = self.in_dataset_series.k3_array
        self.dims_array = self.in_dataset_series.dims_array

        self.is_solve = is_solve
        self.plot_stats = plot_stats

    def launch(self):
        if self.is_solve:
            self.make_solving()

        if self.plot_stats:
            self.make_plot_stats()

    def make_solving(self):
        for dataset_series in self.testing_datasets:
            for dataset in dataset_series:
                base_name = dataset['output_dir'][:len(dataset['output_dir']) - 1]
                for k3 in self.k3_array:
                    kernel = Kernel(k3)
                    dataset['output_dir'] = base_name + '_' + str(int(k3)) + '/'
                    kernel.solve_and_plot([dataset, ])

    def make_plot_stats(self):
        statistics = Statistics(self.testing_datasets, self.dims_array, self.k3_array)
        different_k3_arr = statistics.collect_time_data()
        different_k3_arr_dist, different_k3_arr_wait_time, different_k3_arr_late_time, different_k3_arr_eval = statistics.collect_evaluation()

        plotter = Plot()

        if self.plot_stats == 'time':
            plotter.plot_data(self.testing_datasets, self.dims_array, self.k3_array, different_k3_arr, xlabel='Customers number',
                              ylabel='Time execution, sec')
        elif self.plot_stats == 'distance':
            plotter.plot_data(self.testing_datasets, self.dims_array, self.k3_array, different_k3_arr_dist, xlabel='Customers number',
                              ylabel='Total distance')
        elif self.plot_stats == 'wait_time':
            plotter.plot_data(self.testing_datasets, self.dims_array, self.k3_array, different_k3_arr_wait_time, xlabel='Customers number',
                              ylabel='Wait time')
        elif self.plot_stats == 'late_time':
            plotter.plot_data(self.testing_datasets, self.dims_array, self.k3_array, different_k3_arr_late_time, xlabel='Customers number',
                              ylabel='Late time')
        elif self.plot_stats == 'total_evaluation':
            plotter.plot_data(self.testing_datasets, self.dims_array, self.k3_array, different_k3_arr_eval, xlabel='Customers number',
                              ylabel='Total evaluation')
        else:
            print('Unrecognized plot_stats parameter. Setting it to time...')
            plotter.plot_data(self.testing_datasets, self.dims_array, self.k3_array, different_k3_arr, xlabel='Customers number',
                              ylabel='Time execution, sec')