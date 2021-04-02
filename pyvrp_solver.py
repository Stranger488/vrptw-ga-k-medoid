import numpy as np
import pandas as pd
import pathlib

from time import time

from libs.pyVRP import build_coordinates, build_distance_matrix, genetic_algorithm_vrp, plot_tour_coordinates


class PyVRPSolver:
    def __init__(self, method='tsp'):
        # Parameters - Model
        self.time_window = 'with'  # 'with', 'without'
        self.route = 'closed'  # 'open', 'closed'
        self.model = method  # 'tsp', 'vrp'
        self.graph = True  # True, False

        # Parameters - GA
        self.penalty_value = 10000  # GA Target Function Penalty Value for Violating the Problem Constraints
        self.population_size = 50  # GA Population Size
        self.mutation_rate = 0.10  # GA Mutation Rate
        self.elite = 1  # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained in Each Generation
        self.generations = 10  # GA Number of Generations

    def solve_tsp(self, launch_count, data_dir='cluster_result/'):
        ga_reports = []
        plots_data = []
        for i in range(launch_count):
            coordinates = pd.read_csv('cluster_result/' + data_dir + 'coords{}.txt'.format(i), sep=' ')
            coordinates = coordinates.values
            distance_matrix = build_distance_matrix(coordinates)
            parameters = pd.read_csv('cluster_result/' + data_dir + 'params{}.txt'.format(i), sep=' ')
            parameters = parameters.values

            pathlib.Path('tsptw_result/' + data_dir).mkdir(parents=True, exist_ok=True)

            # Call GA Function

            ga_report, ga_vrp = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, self.population_size,
                                                      self.route, self.model, self.time_window, self.mutation_rate, self.elite,
                                                      self.generations, self.penalty_value, self.graph)

            plot_data = {'coordinates': coordinates, 'ga_vrp': ga_vrp, 'route': self.route}
            plots_data.append(plot_data)

            # Solution Report
            print(ga_report)

            # Save Solution Report
            ga_report.to_csv('tsptw_result/' + data_dir + 'report{}.csv'.format(i), sep=' ', index=False)

            ga_reports.append(ga_report)

        return ga_reports, plots_data

    def solve_vrp(self, points_dataset, tws_all, service_time_all, output_dir='pyvrp_result/'):
        self.generations = 50
        self.elite = 10

        ga_reports = []
        plots_data = []

        parameters = np.concatenate((tws_all, service_time_all), axis=1)
        coordinates = points_dataset
        distance_matrix = build_distance_matrix(coordinates)

        ga_report, ga_vrp = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, self.population_size,
                                                  self.route, self.model, self.time_window, self.mutation_rate,
                                                  self.elite,
                                                  self.generations, self.penalty_value, self.graph)

        plot_data = {'coordinates': coordinates, 'ga_vrp': ga_vrp, 'route': self.route}
        plots_data.append(plot_data)

        ga_reports.append(ga_report)

        # Solution Report
        print(ga_report)

        ga_report.to_csv('pyvrp_result/' + output_dir + 'report.csv', sep=' ', index=False)

        return ga_reports, plots_data
