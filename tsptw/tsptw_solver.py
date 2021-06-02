import pandas as pd
import pathlib
import sys

from multiprocessing import Pool
from functools import partial
from time import time

from tsptw.tsptw_genetic import TSPTWGenetic


class TSPTWSolver:
    def __init__(self, route='closed', graph=False, population_size=50, mutation_rate=0.1, elite=2,
                 generations=20, pool_size=4, k1=10, k2=100):
        # Parameters - Model
        self._route = route  # 'open', 'closed'
        self._graph = graph  # True, False

        # Parameters - evaluation
        self._k1 = k1
        self._k2 = k2

        # Parameters - GA
        self._population_size = population_size  # GA Population Size
        self._mutation_rate = mutation_rate  # GA Mutation Rate
        self._elite = elite  # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained in Each Generation
        self._generations = generations  # GA Number of Generations

        self._pool_size = pool_size
        self._tsptw_genetic = TSPTWGenetic()
        self._BASE_DIR = sys.path[0]

    def _solve(self, launch_count, data_dir='cluster_result/'):
        ga_reports = []
        plots_data = []
        with Pool(self._pool_size) as p:
            func = partial(self._thread_solution, data_dir)
            result = p.map(func, range(launch_count))
            for item in result:
                ga_reports.append(item[1])
                plots_data.append(item[0])
        return ga_reports, plots_data

    def solve_tsp(self, launch_count, data_dir):
        ts = time()
        tsptw_results, plots_data = self._solve(launch_count, data_dir=data_dir)
        te = time()

        output = open(self._BASE_DIR + '/result/tsptw_result/' + data_dir + 'time_tsp.csv', 'w')
        output.write('{}\n'.format(round(te - ts, 4)))
        output.close()

        return tsptw_results, plots_data

    def _thread_solution(self, data_dir, i):
        result = []
        coordinates = pd.read_csv(self._BASE_DIR + '/result/cluster_result/' + data_dir + 'coords{}.txt'.format(i),
                                  sep=' ')
        coordinates = coordinates.values
        distance_matrix = self._tsptw_genetic.build_distance_matrix(coordinates)
        parameters = pd.read_csv(self._BASE_DIR + '/result/cluster_result/' + data_dir + 'params{}.txt'.format(i),
                                 sep=' ')
        parameters = parameters.values

        pathlib.Path(self._BASE_DIR + '/result/tsptw_result/' + data_dir).mkdir(parents=True, exist_ok=True)

        # Call GA Function
        ga_report, ga_vrp = self._tsptw_genetic.genetic_algorithm_tsp(coordinates, distance_matrix, parameters,
                                                                      self._population_size,
                                                                      self._route, self._mutation_rate, self._elite,
                                                                      self._generations, self._graph, self._k1, self._k2)

        plot_data = {'coordinates': coordinates, 'ga_vrp': ga_vrp, 'route': self._route}
        result.append(plot_data)

        # Solution Report
        print(ga_report)

        # Save Solution Report
        ga_report.to_csv(self._BASE_DIR + '/result/tsptw_result/' + data_dir + 'report{}.csv'.format(i), sep=' ',
                         index=False)
        result.append(ga_report)

        return result
