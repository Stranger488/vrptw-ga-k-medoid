import math

import numpy as np

from chromosome import Chromosome
from population import Population


class Solver:
    def __init__(self, Z, distances, P, ng, Pc, Pm):

        self.distances = distances

        # Find k number based on number of customers K and
        # acceptable TSPTW subproblem size Z
        K = len(distances)
        self.k = math.ceil(K / Z)

        self.P = P
        self.ng = ng

        self.Pc = Pc
        self.Pm = Pm

        self.populations = np.array([])
        for _ in range(self.ng):
            self.populations = np.append(self.populations, Population(self.P, self.k, self.distances))

        self.best_chromosome = Chromosome(self.k, self.distances)

    def make_cluster_from_medoids(self):
        medoids = self.best_chromosome.genes
        points_size = int(np.ceil(self.distances[0].size / medoids.size))
        result = np.array([[-1 for _ in range(points_size)] for _ in range(medoids.size)])

        deprecated = np.copy(medoids)

        for i, gene in enumerate(medoids):
            for j in range(points_size):
                if j == 0:
                    result[i][0] = gene
                    continue

                cur_min_ind = -1
                cur_min = math.inf
                for k, el in enumerate(self.distances[gene]):
                    if gene != k and el < cur_min and k not in deprecated:
                        cur_min = el
                        cur_min_ind = k
                result[i][j] = cur_min_ind
                deprecated = np.append(deprecated, cur_min_ind)

        return result

    def solve(self):
        self.populations[0].generate_random_population()
        self.populations[0].calculate_fitness()
        self.save_new_best_chromosome(self.populations[0])

        print("---------")
        print("Iteration {}".format(0))
        print("Best chromosome fitness {}".format(self.best_chromosome.fitness))
        print("Best genes {}".format(self.best_chromosome.genes))
        print("---------")

        for i in range(1, self.ng):
            self.populations[i] = self.populations[i - 1].selection()
            self.populations[i].crossover(self.Pc)
            self.populations[i].mutate(self.Pm)

            print([chromosome.genes for chromosome in self.populations[i].chromosomes])

            self.populations[i].calculate_fitness()
            self.save_new_best_chromosome(self.populations[i])

            print("---------")
            print("Iteration {}".format(i))
            print("Best chromosome fitness {}".format(self.best_chromosome.fitness))
            print("Best genes {}".format(self.best_chromosome.genes))
            print("---------")

        return self.make_cluster_from_medoids()

    def save_new_best_chromosome(self, population):
        chrom_fitness = population.find_best_chromosome().fitness
        if chrom_fitness < self.best_chromosome.fitness:
            self.best_chromosome = population.find_best_chromosome()
