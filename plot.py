import pathlib
import sys
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb2hex


class Plot:
    def __init__(self):

        # Plot parameters
        self._figsize_standart = (8, 6)
        self._dpi_standart = 1200
        self._linewidth_standart = 0.5
        self._width = self.depth = 0.5

        self._BASE_DIR = sys.path[0]

    # Function: Tour Plot
    def plot_tour_coordinates(self, coordinates, solution, axes, color, route):
        depot = solution[0]
        city_tour = solution[1]
        cycol = cycle(
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53', '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c',
             '#aaa662', '#0485d1', '#fffe7a', '#b0dd16', '#85679', '#12e193', '#82cafc', '#ac9362', '#f8481c',
             '#c292a1', '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'])
        # plt.style.use('ggplot')
        for j in range(0, len(city_tour)):
            if (route == 'closed'):
                xy = np.zeros((len(city_tour[j]) + 2, 2))
            else:
                xy = np.zeros((len(city_tour[j]) + 1, 2))
            for i in range(0, xy.shape[0]):
                if (i == 0):
                    xy[i, 0] = coordinates[depot[j][i], 0]
                    xy[i, 1] = coordinates[depot[j][i], 1]
                    if (route == 'closed'):
                        xy[-1, 0] = coordinates[depot[j][i], 0]
                        xy[-1, 1] = coordinates[depot[j][i], 1]
                if (i > 0 and i < len(city_tour[j]) + 1):
                    xy[i, 0] = coordinates[city_tour[j][i - 1], 0]
                    xy[i, 1] = coordinates[city_tour[j][i - 1], 1]
            axes.plot(xy[:, 0], xy[:, 1], 0.0, marker='s', alpha=0.5, markersize=1, color=color, linewidth=0.5)
        for i in range(0, coordinates.shape[0]):
            if i == 0:
                axes.plot(coordinates[i, 0], coordinates[i, 1], 0.0, marker='s', alpha=1.0, markersize=3, color='k')
                # axes.text(coordinates[i,0], coordinates[i,1] + 0.04, z=0.0, s=i,  ha = 'center', va = 'bottom', color = 'k', fontsize = 5)
            else:
                # axes.text(coordinates[i,0],  coordinates[i,1] + 0.04, z=0.0, s=i, ha = 'center', va = 'bottom', color = 'k', fontsize = 5)
                pass
        return

    def plot_clusters(self, init_dataset, dataset, tws, max_tw, depo_spatio, depo_tws, plots_data, text=False):
        plt.rc('font', size=5)  # controls default text sizes
        plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart,
                                 dpi=self._dpi_standart, subplot_kw={'projection': '3d'})

        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')

        axes.set_title('Clusters')

        colors = [rgb2hex([np.random.random_sample(), np.random.random_sample(), np.random.random_sample()])
                  for _ in dataset]

        for i in range(dataset.shape[0]):
            self._plot_with_tws(dataset[i], tws[i], max_tw, colors[i], axes)
            self.plot_tour_coordinates(plots_data[i]['coordinates'], plots_data[i]['ga_vrp'], axes, colors[i],
                                       route=plots_data[i]['route'])

        axes.scatter(depo_spatio[0], depo_spatio[1], 0.0, c='black', s=1)

        axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[0], c='black', s=1)
        axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[1], c='black', s=1)

        axes.bar3d(depo_spatio[0] - self.depth / 8., depo_spatio[1] - self.depth / 8., 0.0, self._width / 4.,
                   self.depth / 4.,
                   max_tw, color='black')

        if text:
            for i, data in enumerate(init_dataset):
                axes.text(data[0], data[1], 0.0, str(i + 1))

        axes.set_zlim(0, None)

    def plot_clusters_parallel(self, init_dataset, dataset, tws, max_tw, depo_spatio, depo_tws, text=False):
        plt.rc('font', size=5)  # controls default text sizes
        plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart,
                                 dpi=self._dpi_standart, subplot_kw={'projection': '3d'})

        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')

        axes.set_title('Clusters')

        colors = [rgb2hex([np.random.random_sample(), np.random.random_sample(), np.random.random_sample()])
                  for _ in dataset]

        for i in range(dataset.shape[0]):
            self._plot_with_tws(dataset[i], tws[i], max_tw, colors[i], axes)

        axes.scatter(depo_spatio[0], depo_spatio[1], 0.0, c='black', s=1)

        axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[0], c='black', s=1)
        axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[1], c='black', s=1)

        axes.bar3d(depo_spatio[0] - self.depth / 8., depo_spatio[1] - self.depth / 8., 0.0, self._width / 4.,
                   self.depth / 4.,
                   max_tw, color='black')

        if text:
            for i, data in enumerate(init_dataset):
                axes.text(data[0], data[1], 0.0, str(i + 1))

        axes.set_zlim(0, None)

    def plot_data(self, testing_datasets, dims_array, k3_array, different_k3_arr, xlabel='x', ylabel='y',
                  mapping=None):
        if mapping is None:
            mapping = ['C', 'R', 'RC']

        for i in range(len(testing_datasets)):
            plt.rc('font', size=5)  # controls default text sizes
            plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=4)
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart, dpi=self._dpi_standart)
            cmap = plt.cm.get_cmap('plasma', 5)

            for j, k3 in enumerate(k3_array):
                self._plot_on_axes(axes, dims_array, different_k3_arr[j][i], c=cmap(j), xlabel=xlabel,
                                   ylabel=ylabel,
                                   label='k3={}'.format(k3),
                                   title='Набор данных {}'.format(mapping[i]))

            axes.grid(True)

            output_dir = str(mapping[i])
            path = self._BASE_DIR + '/result/img/' + output_dir
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            filename = ylabel.replace(',', '')
            fig.savefig(path + '/' + filename.replace(' ', '_'), dpi=self._dpi_standart)

        plt.show()

    def plot_dataset_series_with_bns(self, testing_datasets, dims_array, dataset_arr, bns_arr, xlabel='x', ylabel='y',
                                     mapping=None):
        if mapping is None:
            mapping = ['C', 'R', 'RC']

        for i in range(len(testing_datasets)):
            plt.rc('font', size=5)  # controls default text sizes
            plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=4)
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart, dpi=self._dpi_standart)
            cmap = plt.cm.get_cmap('plasma', 5)

            self._plot_on_axes(axes, dims_array, dataset_arr[i], xlabel=xlabel,
                               ylabel=ylabel, c=cmap(0),
                               label='Результаты работы алгоритма',
                               title='Набор данных {}'.format(mapping[i]))
            self._plot_on_axes(axes, dims_array, bns_arr[i], xlabel=xlabel,
                               ylabel=ylabel, c=cmap(1),
                               label='Наилучшее известное решение',
                               title='Набор данных {}'.format(mapping[i]))

            axes.grid(True)

            output_dir = str(mapping[i])
            path = self._BASE_DIR + '/result/img/' + output_dir
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            filename = ylabel.replace(',', '')
            filename = filename.replace('(', '')
            filename = filename.replace(')', '')
            fig.savefig(path + '/' + filename.replace(' ', '_'), dpi=self._dpi_standart)

        plt.show()

    def plot_dataset_series_hist_with_bns(self, testing_datasets, arr, bns_arr, xlabel='x', ylabel='y',
                                          mapping=None):
        if mapping is None:
            mapping = ['C', 'R', 'RC']

        for i in range(len(testing_datasets)):
            plt.rc('font', size=5)  # controls default text sizes
            plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=4)

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart, dpi=self._dpi_standart)

            axes.hist([arr[i], bns_arr[i]], rwidth=0.7, bins=len(arr[i]), label=['Алгоритм', 'Наилучшее известное решение'])
            # axes.hist(bns_arr, rwidth=0.7, bins=len(testing_datasets[i]))

            axes.grid(True)
            axes.legend(loc='best')

            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)
            axes.set_title('Набор данных {}'.format(mapping[i]))

        plt.show()

    def _plot_with_tws(self, spatial_data, tws, max_tw, colors, axes):
        cluster_size = spatial_data[0].size

        x_data = np.array([i[0] for i in spatial_data])
        y_data = np.array([i[1] for i in spatial_data])

        z_data1 = np.array([i[0] for i in tws])
        z_data2 = np.array([i[1] for i in tws])
        dz_data = np.abs(np.subtract(z_data1, z_data2))

        axes.bar3d(x_data - self.depth / 8., y_data - self.depth / 8., 0.0, self._width / 4., self.depth / 4., max_tw)
        axes.bar3d(x_data - self.depth / 2., y_data - self.depth / 2., z_data1, self._width, self.depth, dz_data)

        axes.scatter(x_data, y_data, 0.0, c=colors, s=cluster_size)
        axes.scatter(x_data, y_data, z_data1, c=colors, s=cluster_size)
        axes.scatter(x_data, y_data, z_data2, c=colors, s=cluster_size)

    def _plot(self, x, y, c='blue', label='label', xlabel='xlabel', ylabel='ylabel'):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart, dpi=self._dpi_standart)

        axes.plot(x, y, color=c, linewidth=self._linewidth_standart, label=label)

        axes.grid()

        axes.legend(loc='best')
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

    def _plot_on_axes(self, axes, x, y, c='green', label='label', xlabel='xlabel', ylabel='ylabel', title='title'):
        axes.plot(x, y, '.-', color=c, linewidth=self._linewidth_standart, label=label)

        axes.grid()

        axes.legend(loc='best')
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title)

    def plot_parallel_time(self, x, y, c='blue', label='label', xlabel='Число процессоров', ylabel='Время выполнения',
                           title='title'):
        plt.rc('font', size=2)  # controls default text sizes
        plt.rc('xtick', labelsize=2)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=2)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=self._dpi_standart)

        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.13, bottom=0.25, right=0.96, top=0.92)

        axes.plot(x, y, '.-', markersize=1, color=c, linewidth=self._linewidth_standart, label=label)

        axes.grid()

        axes.legend(loc='best')
        axes.set_xlabel(xlabel, labelpad=1, fontsize=2)
        axes.set_ylabel(ylabel, labelpad=1, fontsize=2)
        axes.set_title(title, pad=1)

    def show(self):
        plt.show()
