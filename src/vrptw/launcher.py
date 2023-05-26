import importlib

import numpy as np
import pandas as pd

from src.common.plot import Plot
from src.common.statistics import Statistics
from src.common.vrptw_path_holder import VRPTWPathHolder
from src.vrptw.vrptw_solver import VRPTWSolver


class Launcher:
    def __init__(self, launch_entries: str = None,
                 plot_stats: str = None,
                 plot_solutions: str = None,
                 solve_cluster: str = 'default',
                 solve_tsptw: str = 'default'):
        self._launch_entries = importlib.import_module(launch_entries)

        self._vrptw_launch_entry = self._launch_entries.vrptw_launch_entry

        # True, если необходимо собрать статистику по решению
        self._plot_stats = plot_stats

        # True, если необходимо визуализировать итоговое решение
        self._plot_solutions = plot_solutions

        # True, если необходимо запустить первый этап
        self._solve_cluster = solve_cluster

        # True, если необходимо запустить второй этап
        self._solve_tsptw = solve_tsptw

        self._vrptw_path_holder = VRPTWPathHolder(self._vrptw_launch_entry.vrptw_entry_id, solve_cluster, solve_tsptw)

        self._plotter = Plot()

        self.plot_stats_dict = {
            'time_common_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'time_common',
                'xlabel': 'Число клиентов',
                'ylabel': 'Время выполнения программы, с'
            },
            'distance_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'distance',
                'xlabel': 'Число клиентов',
                'ylabel': 'Пройденное расстояние'
            },
            'wait_time_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'wait_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Время ожидания'
            },
            'late_time_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'late_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Время опоздания'
            },
            'total_evaluation_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'total_evaluation',
                'xlabel': 'Число клиентов',
                'ylabel': 'Общая оценка'
            },
            'avg_wait_time_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'avg_wait_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Время ожидания на одно ТС'
            },
            'avg_late_time_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'avg_late_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Время опоздания на одного клиента'
            },
            'max_wait_time_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'max_wait_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Максимальное время ожидания'
            },
            'max_late_time_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'max_late_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Максимальное время опоздания'
            },
            'wait_time_part_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'wait_time_part',
                'xlabel': 'Число клиентов',
                'ylabel': 'Доля ожиданий среди всех вершин'
            },
            'late_time_part_stats': {
                'plot_lambda': self._plotter.plot_stats,
                'data_column_name': 'late_time_part',
                'xlabel': 'Число клиентов',
                'ylabel': 'Доля опозданий среди всех вершин'
            },
            'distance_bks_stats': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'data_column_name': 'distance',
                'xlabel': 'Число клиентов',
                'ylabel': 'Пройденное расстояние (сравнение с наилучшими решениями)'
            },
            'wait_time_bks_stats': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'data_column_name': 'wait_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Время ожидания (сравнение с наилучшими решениями)'
            },
            'late_time_bks_stats': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'data_column_name': 'late_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Время опоздания (сравнение с наилучшими решенияим)'
            },
            'total_time_bks_stats': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'data_column_name': 'total_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Общее время в пути (сравнение с наилучшими решенияим)'
            },
            'avg_wait_time_bks_stats': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'data_column_name': 'avg_wait_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Время ожидания на одно ТС (сравнение с наилучшими решениями)'
            },
            'avg_late_time_bks_stats': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'data_column_name': 'avg_late_time',
                'xlabel': 'Число клиентов',
                'ylabel': 'Время опоздания на одного клиента (сравнение с наилучшими решенияим)'
            },
            'std_wait_time_bks_stats': {
                'plot_lambda': self._plotter.plot_stats_bks_hist,
                'data_column_name': 'std_wait_time',
                'xlabel': '',
                'ylabel': 'Среднеквадратичное отклонение по времени ожидания (сравнение с наилучшими решенияим)'
            }
        }

    def launch(self):
        self._make_solving()

        if self._plot_solutions:
            self._make_plot_solutions()

        if self._plot_stats:
            self._make_plot_stats()

    def _make_solving(self):
        vrptw_solver = VRPTWSolver(self._vrptw_launch_entry, self._vrptw_path_holder)
        vrptw_solver.solve(self._solve_cluster, self._solve_tsptw)

    def _make_plot_stats(self):
        statistics = Statistics(self._vrptw_launch_entry, self._vrptw_path_holder)

        stats_df = statistics.collect_all_stats()
        bks_stats_df = statistics.collect_bks_stats()

        for stat in self._vrptw_launch_entry.plot_stats_type_arr:
            plot_lambda = self.plot_stats_dict[stat]['plot_lambda']

            plot_lambda(stats_df, bks_stats_df, self.plot_stats_dict[stat]['data_column_name'],
                        xlabel=self.plot_stats_dict[stat]['xlabel'],
                        ylabel=self.plot_stats_dict[stat]['ylabel'],
                        output_dir=self._vrptw_path_holder.PLOT_STATS_OUTPUT)

    def _make_plot_solutions(self):
        for entry in self._vrptw_launch_entry.cluster_launch_entry_arr:
            data = pd.read_fwf(self._vrptw_path_holder.BASE_DIR + '/input/task/'
                               + entry.dataset.data_file)
            vehicle_number = int(data['VEHICLE_NUMBER'][0])

            coords, params, report = VRPTWSolver.read_input_for_plot_solutions(vehicle_number,
                                                                               self._vrptw_path_holder.CLUSTER_OUTPUT
                                                                               + entry.common_id + '/',
                                                                               self._vrptw_path_holder.TSPTW_OUTPUT
                                                                               + entry.common_id + '/')
            coords_sorted = np.empty(vehicle_number, dtype=object)
            params_sorted = np.empty(vehicle_number, dtype=object)
            for v in range(vehicle_number):
                # v-е ТС, на второй позиции индексы вершин внутри кластера,
                # slice до -3 чтобы получить индексы без возвращения в депо
                permutation_list = [int(el) for el in report[v]['Job'][:-3]]
                coords_sorted[v] = np.array([coords[v][i] for i in permutation_list])
                params_sorted[v] = np.array([params[v][i] for i in permutation_list])

            # Можно в будущем настроить параметр запуска, но для удобной визулиации лучше без возвращения ТС в депо
            # route_type = 'closed' if report[0][2].values[0] == report[0][2].values[-3] else 'open'
            route_type = 'open'
            self._plotter.plot_clusters_routes(coords_sorted, params_sorted,
                                               route_type, self._vrptw_launch_entry.is_text,
                                               self._vrptw_path_holder.PLOT_SOLUTIONS_OUTPUT
                                               + entry.common_id)

        # self._plotter.show()
