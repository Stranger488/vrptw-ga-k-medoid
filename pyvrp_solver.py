import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

from libs.pyVRP import build_coordinates, build_distance_matrix, genetic_algorithm_vrp, plot_tour_coordinates\

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
        self.generations = 15  # GA Number of Generations
   
    def thread_solution(self, data_dir, i):
        result = []
        coordinates = pd.read_csv('cluster_result/' + data_dir + 'coords{}.txt'.format(i), sep=' ')
        coordinates = coordinates.values
        distance_matrix = build_distance_matrix(coordinates)
        parameters = pd.read_csv('cluster_result/' + data_dir + 'params{}.txt'.format(i), sep=' ')
        parameters = parameters.values

        # Call GA Function
        ga_report, ga_vrp = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, self.population_size,
                                                self.route, self.model, self.time_window, self.mutation_rate, self.elite,
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
    
        # ga_reports = []
        # plots_data = []
        # for i in range(launch_count):
        #     if threading.active_count() < 6:
        #         th[i] = Thread(target=thread_solution, args=(i, ))
        #         th[i].start()
        #     if i == in range(6):
        #         if (!th[i])
        
        # i_arr = np.arange(0, launch_count, 1)
        # print(i_arr)
        ga_reports = []
        plots_data = []
        with Pool(4) as p:
            func = partial(self.thread_solution, data_dir)
            result = p.map(func, range(launch_count))
            for item in result:
                ga_reports.append(item[1])
                plots_data.append(item[0])
        return ga_reports, plots_data