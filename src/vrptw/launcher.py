import importlib

from src.common.plot import Plot
from src.common.statistics import Statistics
from src.vrptw.vrptw_solver import VRPTWSolver


class Launcher:
    def __init__(self, launch_entries: str = None,
                 plot_stats: str = None,
                 solve_cluster: str = 'default',
                 solve_tsptw: str = 'default'):
        self._launch_entries = importlib.import_module(launch_entries)

        self._vrptw_launch_entry = self._launch_entries.vrptw_launch_entry

        # True, если необходимо после решения собрать статистику
        self._plot_stats = plot_stats

        # True, если необходимо запустить первый этап
        self._solve_cluster = solve_cluster

        # True, если необходимо запустить второй этап
        self._solve_tsptw = solve_tsptw

        self._plotter = Plot()

        self.plot_stats_dict = {
            'time_common': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Время выполнения программы, с'
            },
            'distance': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Пройденное расстояние'
            },
            'wait_time': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Время ожидания'
            },
            'late_time': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Время опоздания'
            },
            'total_evaluation': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Общая оценка'
            },
            'avg_wait_time': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Время ожидания на одно ТС'
            },
            'avg_late_time': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Время опоздания на одного клиента'
            },
            'max_wait_time': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Максимальное время ожидания'
            },
            'max_late_time': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Максимальное время опоздания'
            },
            'wait_time_part': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Доля ожиданий среди всех вершин'
            },
            'late_time_part': {
                'plot_lambda': self._plotter.plot_stats,
                'xlabel': 'Число клиентов',
                'ylabel': 'Доля опозданий среди всех вершин'
            },
            'distance_bks': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'xlabel': 'Число клиентов',
                'ylabel': 'Пройденное расстояние (сравнение с наилучшими решениями)'
            },
            'wait_time_bks': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'xlabel': 'Число клиентов',
                'ylabel': 'Время ожидания (сравнение с наилучшими решениями)'
            },
            'late_time_bks': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'xlabel': 'Число клиентов',
                'ylabel': 'Время опоздания (сравнение с наилучшими решенияим)'
            },
            'total_time_bks': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'xlabel': 'Число клиентов',
                'ylabel': 'Общее время в пути(сравнение с наилучшими решенияим)'
            },
            'avg_wait_time_bks': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'xlabel': 'Число клиентов',
                'ylabel': 'Время ожидания на одно ТС (сравнение с наилучшими решениями)'
            },
            'avg_late_time_bks': {
                'plot_lambda': self._plotter.plot_stats_bks,
                'xlabel': 'Число клиентов',
                'ylabel': 'Время опоздания на одного клиента (сравнение с наилучшими решенияим)'
            },
            'std_wait_time_bks': {
                'xlabel': 'Число клиентов',
                'ylabel': 'Среднеквадратичное отклонение по времени ожидания (сравнение с наилучшими решенияим)'
            }
        }

    def launch(self):
        self._make_solving()

        if self._plot_stats:
            self._make_plot_stats()

    def _make_solving(self):
        vrptw_solver = VRPTWSolver(vrptw_launch_entry=self._vrptw_launch_entry)
        vrptw_solver.solve_and_plot(self._solve_cluster, self._solve_tsptw)

    def _make_plot_stats(self):
        statistics = Statistics(self._vrptw_launch_entry)

        stats_df = statistics.collect_all_stats()
        bks_stats_df = statistics.collect_bks_stats()

        # TODO: bks plot
        for stat in self._vrptw_launch_entry.plot_stats_type_arr:
            plot_lambda = self.plot_stats_dict[stat]['plot_lambda']

            plot_lambda(stats_df, bks_stats_df, stat, xlabel=self.plot_stats_dict[stat]['xlabel'],
                        ylabel=self.plot_stats_dict[stat]['ylabel'],
                        output_dir=self._vrptw_launch_entry.PLOT_STATS_OUTPUT)
