import numpy as np
import pandas as pd
import pathlib
from multiprocessing import Pool
from functools import partial
from time import time

from libs.pyVRP import build_coordinates, build_distance_matrix, genetic_algorithm_tsp, plot_tour_coordinates


class PyVRPSolver:
    def __init__(self):
        self.pool_size = 4

        # Parameters - Model
        self.route = 'closed'  # 'open', 'closed'
        self.graph = True  # True, False

        # Parameters - GA
        self.penalty_value = 1000  # GA Target Function Penalty Value for Violating the Problem Constraints
        self.population_size = 50  # GA Population Size
        self.mutation_rate = 0.10  # GA Mutation Rate
        self.elite = 2  # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained in Each Generation
        self.generations = 20  # GA Number of Generations

    def thread_solution(self, data_dir, i):
        result = []
        coordinates = pd.read_csv('cluster_result/' + data_dir + 'coords{}.txt'.format(i), sep=' ')
        coordinates = coordinates.values
        distance_matrix = build_distance_matrix(coordinates)
        parameters = pd.read_csv('cluster_result/' + data_dir + 'params{}.txt'.format(i), sep=' ')
        parameters = parameters.values

        pathlib.Path('tsptw_result/' + data_dir).mkdir(parents=True, exist_ok=True)

        # Call GA Function
        ga_report, ga_vrp = genetic_algorithm_tsp(coordinates, distance_matrix, parameters, self.population_size,
                                                self.route, self.mutation_rate, self.elite,
                                                self.generations, self.penalty_value, self.graph)

        plot_data = {'coordinates': coordinates, 'ga_vrp': ga_vrp, 'route': self.route}
        result.append(plot_data)

        # Solution Report
        print(ga_report)

        # Save Solution Report
        ga_report.to_csv('tsptw_result/' + data_dir + 'report{}.csv'.format(i), sep=' ', index=False)
        result.append(ga_report)

        return result

    def solve_tsp(self, launch_count, data_dir='cluster_result/'):
        ga_reports = []
        plots_data = []
        with Pool(self.pool_size) as p:
            func = partial(self.thread_solution, data_dir)
            result = p.map(func, range(launch_count))
            for item in result:
                ga_reports.append(item[1])
                plots_data.append(item[0])
        return ga_reports, plots_data
        