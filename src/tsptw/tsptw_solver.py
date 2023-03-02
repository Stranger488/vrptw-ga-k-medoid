import os
import pathlib
from functools import partial
from multiprocessing import Pool

import pandas as pd

from src.tsptw.tsptw_genetic import TSPTWGenetic


class TSPTWSolver:
    def __init__(self, vehicle_number, coordinates, distance_matrix, parameters,
                 route='closed', population_size=50, mutation_rate=0.1, elite=2,
                 generations=20, pool_size=4, k1=10, k2=100):
        # Read data
        self._vehicle_number = vehicle_number
        self._coordinates = coordinates
        self._distance_matrix = distance_matrix
        self._parameters = parameters
        # Parameters - Model
        self._route = route  # 'open', 'closed'

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
        self._BASE_DIR = os.path.abspath(os.curdir)

    def solve_tsptw_parallel(self):
        ga_reports = []
        plots_data = []
        with Pool(self._pool_size) as p:
            result = p.map(self._thread_solution, range(self._vehicle_number))
            for item in result:
                ga_reports.append(item[1])
                plots_data.append(item[0])
        return ga_reports, plots_data

    def solve_tsptw_core_data_mining(self):
        pass

    def solve(self):
        ga_reports = []
        plots_data = []
        with Pool(self._pool_size) as p:
            func = partial(self._thread_solution, data_dir)
            result = p.map(func, range(launch_count))
            for item in result:
                ga_reports.append(item[1])
                plots_data.append(item[0])
        return ga_reports, plots_data

    def _thread_solution(self, i):
        result = []
        # Call GA Function
        ga_report, ga_vrp = self._tsptw_genetic.genetic_algorithm_tsp(self._coordinates[i], self._distance_matrix[i], self._parameters[i],
                                                                      self._population_size,
                                                                      self._route, self._mutation_rate, self._elite,
                                                                      self._generations, self._k1, self._k2)

        plot_data = {'coordinates': self._coordinates, 'ga_vrp': ga_vrp, 'route': self._route}
        result.append(plot_data)

        # Solution Report
        print(ga_report)

        result.append(ga_report)
        return result