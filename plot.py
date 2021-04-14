import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import rgb2hex
from libs.pyVRP import plot_tour_coordinates


class Plot:
    def __init__(self):

        # Plot parameters
        self.figsize_standart = (8, 6)
        self.dpi_standart = 400
        self.linewidth_standart = 0.5
        self.width = self.depth = 0.5

    def plot_clusters(self, init_dataset, dataset, tws, max_tw, depo_spatio, depo_tws, plots_data, text=False):
        plt.rc('font', size=5)  # controls default text sizes
        plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self.figsize_standart,
                                 dpi=self.dpi_standart, subplot_kw={'projection': '3d'})

        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')

        axes.set_title('Clusters')

        colors = [rgb2hex([np.random.random_sample(), np.random.random_sample(), np.random.random_sample()])
                  for _ in dataset]

        for i in range(dataset.shape[0]):
            self.plot_with_tws(dataset[i], tws[i], max_tw, colors[i], axes)
            plot_tour_coordinates(plots_data[i]['coordinates'], plots_data[i]['ga_vrp'], axes, colors[i],
                                  route=plots_data[i]['route'])

        axes.scatter(depo_spatio[0], depo_spatio[1], 0.0, c='black', s=1)

        axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[0], c='black', s=1)
        axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[1], c='black', s=1)

        axes.bar3d(depo_spatio[0] - self.depth / 8., depo_spatio[1] - self.depth / 8., 0.0, self.width / 4., self.depth / 4.,
                   max_tw, color='black')

        if text:
            for i, data in enumerate(init_dataset):
                axes.text(data[0], data[1], 0.0, str(i + 1))

        axes.set_zlim(0, None)

    def plot_with_tws(self, spatial_data, tws, max_tw, colors, axes):
        cluster_size = spatial_data[0].size

        x_data = np.array([i[0] for i in spatial_data])
        y_data = np.array([i[1] for i in spatial_data])

        z_data1 = np.array([i[0] for i in tws])
        z_data2 = np.array([i[1] for i in tws])
        dz_data = np.abs(np.subtract(z_data1, z_data2))

        axes.bar3d(x_data - self.depth / 8., y_data - self.depth / 8., 0.0, self.width / 4., self.depth / 4., max_tw)
        axes.bar3d(x_data - self.depth / 2., y_data - self.depth / 2., z_data1, self.width, self.depth, dz_data)

        axes.scatter(x_data, y_data, 0.0, c=colors, s=cluster_size)
        axes.scatter(x_data, y_data, z_data1, c=colors, s=cluster_size)
        axes.scatter(x_data, y_data, z_data2, c=colors, s=cluster_size)

    def plot(self, x, y, c='blue', label='label', xlabel='xlabel', ylabel='ylabel'):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self.figsize_standart, dpi=self.dpi_standart)

        axes.plot(x, y, color=c, linewidth=self.linewidth_standart, label=label)

        axes.grid()

        axes.legend(loc='best')
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

    def plot_on_axes(self, axes, x, y, c='green', label='label', xlabel='xlabel', ylabel='ylabel', title='title'):
        axes.plot(x, y, '.-', color=c, linewidth=self.linewidth_standart, label=label)

        axes.grid()

        axes.legend(loc='best')
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title)

    def plot_data(self, testing_datasets, dims_array, k3_array, different_k3_arr, xlabel='x', ylabel='y', mapping=None):
        if mapping is None:
            mapping = ['C', 'R', 'RC']

        for i in range(len(testing_datasets)):
            plt.rc('font', size=5)  # controls default text sizes
            plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=4)
            _, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 15), dpi=400)
            cmap = plt.cm.get_cmap('plasma', 5)

            for j, k3 in enumerate(k3_array):
                self.plot_on_axes(axes, dims_array, different_k3_arr[j][i], c=cmap(j), xlabel=xlabel,
                                     ylabel=ylabel,
                                     label='{}'.format(k3), title='{} dataset series'.format(mapping[i]))

            axes.grid()

        plt.show()
