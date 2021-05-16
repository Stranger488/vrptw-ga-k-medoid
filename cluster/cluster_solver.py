import math
import sys

import numpy as np

from time import time

from cluster.chromosome import Chromosome
from cluster.population import Population


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

        self._current_population = Population(self._P, self._k, self._distances)
        self._best_chromosome = Chromosome(self._k, self._distances)

        self._numpy_random = numpy_rand

        self._BASE_DIR = sys.path[0]

    def solve_cluster(self, output_dir):
        ts = time()
        result = self._solve()
        te = time()

        output = open(self._BASE_DIR + '/result/cluster_result/' + output_dir + 'time_cluster.csv', 'w')
        output.write('{}\n'.format(round(te - ts, 4)))
        output.close()

        return result

    def _solve(self):
        self._current_population.generate_random_population(self._numpy_random)
        self._current_population.calculate_fitness()
        self._save_new_best_chromosome(self._current_population)

        print("---------")
        print("Iteration {}".format(0))
        print("Best chromosome fitness {}".format(self._best_chromosome.fitness))
        print("All chromosomes genes: {}".format(
            [chromosome.genes.tolist() for chromosome in self._current_population.chromosomes]))
        print("Best genes {}".format(self._best_chromosome.genes))
        print("---------")

        for i in range(1, self._ng):
            self._current_population = self._current_population.selection(self._numpy_random)
            self._current_population.crossover(self._Pc, self._Pmb, self._numpy_random)
            self._current_population.mutate(self._Pm, self._numpy_random)

            self._current_population.calculate_fitness()
            self._save_new_best_chromosome(self._current_population)

            print("---------")
            print("Iteration: {}".format(i))
            print("Best chromosome fitness: {}".format(self._best_chromosome.fitness))
            print("All chromosomes genes: {}".format(
                [chromosome.genes.tolist() for chromosome in self._current_population.chromosomes]))
            print("Best genes: {}".format(self._best_chromosome.genes))
            print("---------")

        return self._make_cluster_from_medoids()

    def _save_new_best_chromosome(self, population):
        chrom_fitness = population.find_best_chromosome().fitness
        print("cur fitness: {}".format(chrom_fitness))
        if chrom_fitness < self._best_chromosome.fitness:
            self._best_chromosome = population.find_best_chromosome()

    def _make_cluster_from_medoids(self):
        medoids = self._best_chromosome.genes
        points_size = int(np.ceil(self._distances[0].size / medoids.size))
        result = np.array([[-1 for _ in range(points_size)] for _ in range(medoids.size)])

        deprecated = np.copy(medoids)

        for i, gene in enumerate(medoids):
            for j in range(points_size):
                if j == 0:
                    result[i][0] = gene
                    continue

                cur_min_ind = -1
                cur_min = math.inf
                for k, el in enumerate(self._distances[gene]):
                    if gene != k and el < cur_min and k not in deprecated:
                        cur_min = el
                        cur_min_ind = k
                result[i][j] = cur_min_ind
                deprecated = np.append(deprecated, cur_min_ind)

        return result

