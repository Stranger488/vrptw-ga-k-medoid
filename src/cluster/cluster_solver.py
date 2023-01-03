import math
import os

import numpy as np

from src.cluster.chromosome import Chromosome
from src.cluster.population import Population


class ClusterSolver:
    def __init__(self, distances, Z=10, P=20, ng=25, Pc=0.65, Pm=0.2, Pmb=0.05, k=None, numpy_rand=None):
        self._distances = distances

        # Find k number based on number of customers K and
        # acceptable TSPTW subproblem size Z
        if k is None:
            K = len(distances)
            self._k = math.ceil(K / Z)
        else:
            self._k = k

        self._P = P
        self._ng = ng

        self._Pc = Pc
        self._Pm = Pm
        self._Pmb = Pmb

        self._numpy_rand = np.random.RandomState(42)

        self._current_population = Population(self._P, self._k, self._distances)
        self._best_chromosome = Chromosome(self._k, self._distances)

        self._BASE_DIR = os.path.abspath(os.curdir)

    # параллелизм здесь внутри будет для режима data_mining
    def solve_cluster_core_data_mining(self):
        pass

    def solve(self):
        self._current_population.generate_random_population(self._numpy_rand)
        self._current_population.calculate_fitness()
        self._save_new_best_chromosome(self._current_population)

        self.print_current_iteration_info(0)
        for i in range(1, self._ng):
            self._current_population = self._current_population.selection(self._numpy_rand)
            self._current_population.crossover(self._Pc, self._Pmb, self._numpy_rand)
            self._current_population.mutate(self._Pm, self._numpy_rand)

            self._current_population.calculate_fitness()
            self._save_new_best_chromosome(self._current_population)

            self.print_current_iteration_info(i)

        return self._make_cluster_from_medoids()

    def print_current_iteration_info(self, i):
        print("---------")
        print("Iteration: {}".format(i))
        print("Best chromosome fitness: {}".format(self._best_chromosome.fitness))
        print("All chromosomes genes: {}".format(
            [chromosome.genes.tolist() for chromosome in self._current_population.chromosomes]))
        print("Best genes: {}".format(self._best_chromosome.genes))
        print("---------")

    def _save_new_best_chromosome(self, population):
        best_chromosome_ind = population.find_best_chromosome_ind()
        best_chromosome = population.chromosomes[best_chromosome_ind]
        print("cur fitness: {}".format(best_chromosome.fitness))
        if best_chromosome.fitness < self._best_chromosome.fitness:
            self._best_chromosome = best_chromosome

    # TODO: проанализировать, в последние маршруты попадают одни и те же вершины
    def _make_cluster_from_medoids(self):
        medoids = self._best_chromosome.genes
        points_size = int(np.ceil(self._distances[0].size / medoids.size))
        result = np.full((medoids.size, points_size), -1)

        approved = np.arange(self._distances[0].size)

        # Убрать медоиды из поиска, вручную их помещаем в кластер на нулевую позицию
        approved = np.delete(approved, np.ravel([np.where(approved == med) for med in medoids]))

        for i, gene in enumerate(medoids):
            # Сразу помещаем медоид
            result[i][0] = gene
            for j in range(1, points_size):
                if approved.size != 0:
                    # Строка с расстояниями до других вершин
                    cur_dist = self._distances[gene]

                    # Ищем индекс ближайшей вершины
                    cur_min_ind = np.ravel(np.where(cur_dist == cur_dist[approved].min()))[0]
                    result[i][j] = cur_min_ind

                    # Удаляем из списка разрешенных
                    approved = approved[approved != cur_min_ind]

        return result
