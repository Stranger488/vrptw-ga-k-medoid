# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb2hex

from src.common.utils import create_directory


# matplotlib.use('TkAgg')


class Plot:
    def __init__(self):

        # Plot parameters
        self._figsize_standart = (8, 6)
        self._dpi_standart = 1200
        self._linewidth_standart = 0.5
        self._width = self.depth = 0.5

    def _plot_route(self, spatial_data, color, route_type, axes):
        # Пары (x, y)
        plot_data = np.copy(spatial_data)
        depot = np.copy(spatial_data[0])
        if route_type == 'closed':
            plot_data = np.append(plot_data, depot.reshape((1, 2)), axis=0)

        # Вывести все маршруты
        axes.plot(plot_data[:, 0], plot_data[:, 1], 0.0, marker='s', alpha=0.5,
                  markersize=1, color=color, linewidth=0.5)

    def plot_clusters_routes(self, points_with_ind, tws, route_type, text, output_dir):
        plt.rc('font', size=5)  # controls default text sizes
        plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart,
                                 dpi=self._dpi_standart, subplot_kw={'projection': '3d'})

        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')

        axes.set_title('Кластеры и маршруты')

        colors = [rgb2hex([np.random.random_sample(), np.random.random_sample(), np.random.random_sample()])
                  for _ in points_with_ind]

        max_tw = tws[0][0][1]

        # Для каждого ТС
        for v in range(points_with_ind.shape[0]):
            self._plot_tws(points_with_ind[v][1:, :2], tws[v][1:], max_tw, colors[v], axes)
            self._plot_route(points_with_ind[v][:, :2], colors[v], route_type, axes)

            if text:
                for p in points_with_ind[v]:
                    axes.text(p[0], p[1], 0.0, str(int(p[2])))

        # Берем у первого ТС информацию о депо
        first_points = points_with_ind[0]
        first_tws = tws[0]
        axes.scatter(first_points[0, 0], first_points[0, 1], 0.0, c='black', s=1)
        axes.scatter(first_points[0, 0], first_points[0, 1], first_tws[0, 1], c='black', s=1)
        axes.scatter(first_points[0, 0], first_points[0, 1], first_tws[0, 1], c='black', s=1)

        axes.bar3d(first_points[0, 0] - self.depth / 8., first_points[0, 1] - self.depth / 8., 0.0, self._width / 4.,
                   self.depth / 4., max_tw, color='black')

        # Для депо выводим еще дополнительную метку
        axes.plot(first_points[0, 0], first_points[0, 1], 0.0, marker='s', alpha=1.0, markersize=3, color='k')

        create_directory(output_dir)
        fig.savefig(output_dir + '/solution', dpi=self._dpi_standart)

    def plot_stats(self, stats_df, bks_stats_df, data_column_name, xlabel='x', ylabel='y', output_dir=''):
        grouped = stats_df.groupby('dataset_type')
        for i, (name, group) in enumerate(grouped):
            k3_grouped = group.groupby('k3')
            plt.rc('font', size=5)  # controls default text sizes
            plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=4)
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart, dpi=self._dpi_standart)
            cmap = plt.cm.get_cmap('plasma', group.shape[0])
            for j, (k3, g) in enumerate(k3_grouped):
                self._plot_on_axes(axes, g['dim'], g[data_column_name], xlabel=xlabel,
                                   ylabel=ylabel, c=cmap(j),
                                   label='Результаты работы алгоритма, k3 = {}'.format(k3),
                                   title='Набор данных {}'.format(name))

            axes.grid(True)

            final_output_dir = output_dir + name
            create_directory(final_output_dir)
            filename = ylabel.replace(',', '').replace(' ', '_')
            fig.savefig(final_output_dir + '/' + filename, dpi=self._dpi_standart)
            plt.close(fig)
        # plt.show()

    def plot_stats_bks(self, stats_df, bks_stats_df, data_column_name, xlabel='x', ylabel='y', output_dir=''):
        grouped = stats_df.groupby('dataset_type')
        for i, (name, group) in enumerate(grouped):
            k3_grouped = group.groupby('k3')
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart, dpi=self._dpi_standart)
            cmap = plt.cm.get_cmap('plasma', group.shape[0] + 1)
            plt.rc('font', size=5)  # controls default text sizes
            plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=4)
            for j, (k3, g) in enumerate(k3_grouped):
                self._plot_on_axes(axes, g['dim'], g[data_column_name], xlabel=xlabel,
                                   ylabel=ylabel, c=cmap(j),
                                   label='Результаты работы алгоритма, k3 = {}'.format(k3),
                                   title='Набор данных {}'.format(name))

                if j == k3_grouped.ngroups - 1:
                    self._plot_on_axes(axes, g['dim'],
                                       bks_stats_df[bks_stats_df['name'].isin(g['name'].values)].sort_values(
                                           'wait_time')[data_column_name],
                                       xlabel=xlabel,
                                       ylabel=ylabel, c=cmap(group.shape[0] + 1),
                                       label='Наилучшее известное решение',
                                       title='Набор данных {}'.format(name))
            axes.grid(True)

            final_output_dir = output_dir + name
            create_directory(final_output_dir)
            filename = ylabel.replace(',', '').replace('(', '').replace(')', '').replace(' ', '_')
            fig.savefig(final_output_dir + '/' + filename, dpi=self._dpi_standart)

        # plt.show()

    def plot_stats_bks_hist(self, stats_df, bks_stats_df, data_column_name, xlabel='x', ylabel='y', output_dir=''):
        grouped = stats_df.groupby('dataset_type')
        for i, (name, group) in enumerate(grouped):
            k3_grouped = group.groupby('k3')
            plt.rc('font', size=5)  # controls default text sizes
            plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=4)
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart, dpi=self._dpi_standart)
            for j, (k3, g) in enumerate(k3_grouped):
                axes.hist(
                    g[data_column_name],
                    rwidth=1.0,
                    bins=len(g[data_column_name]),
                    label='Результаты работы алгоритма, k3 = {}'.format(k3)
                )

            bks_hist_data = bks_stats_df[bks_stats_df['name'] == group['name'].iloc[0]][data_column_name]
            axes.hist(
                bks_hist_data,
                rwidth=0.7,
                bins=len(bks_hist_data),
                label='Наилучшее известное решение'
            )

            axes.grid(True)
            axes.legend(loc='best')
            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)
            axes.set_title('Набор данных {}'.format(name))

            final_output_dir = output_dir + name
            create_directory(final_output_dir)
            filename = ylabel.replace(',', '').replace('(', '').replace(')', '').replace(' ', '_')
            fig.savefig(final_output_dir + '/' + filename, dpi=self._dpi_standart)
            plt.close(fig)

        # plt.show()

    def _plot_tws(self, spatial_data, tws, max_tw, colors, axes):
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

    def _plot_on_axes(self, axes, x, y, c='green', label='label', xlabel='xlabel', ylabel='ylabel', title='title',
                      linestyle='-.', linewidth=0.5):
        axes.plot(x, y, '.-', color=c, linewidth=linewidth, label=label)

        axes.grid()

        axes.legend(loc='best')
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title)

    # TODO: доработать
    def plot_parallel_time(self, x, y, c='blue', label='Зависимость времени выполнения от числа процессоров',
                           xlabel='Число процессоров', ylabel='Время выполнения',
                           title=''):
        plt.rc('font', size=5)  # controls default text sizes
        plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=4)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=self._dpi_standart)
        axes.plot(x, y, '.-', markersize=3, color=c, linewidth=self._linewidth_standart, label=label)

        axes.grid()

        axes.legend(loc='best')
        axes.set_xlabel(xlabel, labelpad=1, fontsize=4)
        axes.set_ylabel(ylabel, labelpad=1, fontsize=4)
        axes.set_title(title, pad=1)

        fig.savefig(fname='test.png', dpi=self._dpi_standart)
        plt.show()

    def plot_stats_function(self, stats_df, bks_stats_df, data_column_name, xlabel='x', ylabel='y', output_dir=''):
        grouped = stats_df.groupby('dataset_type')
        for i, (name, group) in enumerate(grouped):
            dim_grouped = group.groupby('dim')

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart, dpi=self._dpi_standart)
            cmap = plt.cm.get_cmap('plasma', group.shape[0] + 1)
            plt.rc('font', size=5)  # controls default text sizes
            plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=4)

            for j, (dim, gr) in enumerate(dim_grouped):
                g = gr.loc[group['k3'] == 100]

                list_values = g[data_column_name].values.tolist()[0]
                inds = [ind for ind in range(len(list_values))]
                self._plot_on_axes(axes, inds, list_values, xlabel=xlabel,
                                   ylabel=ylabel, c=cmap(j),
                                   label='График сходимости целевой функции, размерность {}'.format(dim),
                                   title='Набор данных {}'.format(name),
                                   linestyle='-',
                                   linewidth=1)
            axes.grid(True)

            final_output_dir = output_dir + name
            create_directory(final_output_dir)
            filename = ylabel.replace(',', '').replace(' ', '_')
            fig.savefig(final_output_dir + '/' + filename, dpi=self._dpi_standart)
            plt.close(fig)

    def plot_stats_all(self, stats_df, bks_stats_df, data_column_name, xlabel='x', ylabel='y', output_dir=''):
        grouped = stats_df.groupby('dataset_type')
        for i, (name, group) in enumerate(grouped):
            g = group.loc[group['k3'] == 100]
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self._figsize_standart, dpi=self._dpi_standart)
            cmap = plt.cm.get_cmap('plasma', group.shape[0] + 1)
            plt.rc('font', size=5)  # controls default text sizes
            plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=4)

            self._plot_on_axes(axes, g['dim'], g[data_column_name], xlabel=xlabel,
                               ylabel=ylabel, c='green',
                               label='Результаты работы разработанного ранее алгоритма',
                               title='Набор данных {}'.format(name))

            new_column_name = data_column_name.replace('_time_', '_time_old_')
            self._plot_on_axes(axes, g['dim'], g[new_column_name], xlabel=xlabel,
                               ylabel=ylabel, c='red',
                               label='Результаты работы модифицированного алгоритма',
                               title='Набор данных {}'.format(name))

            self._plot_on_axes(axes, g['dim'],
                               bks_stats_df[bks_stats_df['name'].isin(g['name'].values)].sort_values(
                                   'wait_time')[data_column_name],
                               xlabel=xlabel,
                               ylabel=ylabel, c='blue',
                               label='Наилучшее известное решение',
                               title='Набор данных {}'.format(name))
            axes.grid(True)

            final_output_dir = output_dir + name
            create_directory(final_output_dir)
            filename = ylabel.replace(',', '').replace('(', '').replace(')', '').replace(' ', '_')
            fig.savefig(final_output_dir + '/' + filename, dpi=self._dpi_standart)

    @staticmethod
    def show():
        plt.show()
