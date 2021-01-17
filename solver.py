import math

import numpy as np

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

        self.populations = np.empty(self.ng, dtype=Population)
        self.populations.fill(Population(self.P, self.k, self.distances))

    def make_cluster_from_medoids(self):
        medoids = self.populations[self.ng - 1].chromosomes[self.populations[self.ng - 1].best_chromosome_ind].genes

        points_size = int(np.ceil(self.distances[0].size / medoids.size))

        result = np.array([[-1 for _ in range(points_size)] for _ in range(medoids.size)])

        for i, gene in enumerate(medoids):
            for j in range(points_size):
                if j == 0:
                    result[i][0] = gene
                    continue

                cur_min_ind = -1
                cur_min = math.inf
                for k, el in enumerate(self.distances[gene]):
                    if gene != k and el < cur_min and k not in result:
                        cur_min = el
                        cur_min_ind = k
                result[i][j] = cur_min_ind

        return result

    def solve(self):

        self.populations[0].generate_random_population()
        self.populations[0].calculate_fitness()

        print(self.populations[0].chromosomes[0].fitness)
        print(self.populations[0].chromosomes[1].fitness)

        for i in range(1, self.ng):

            self.populations[i] = self.populations[i - 1].selection()
            self.populations[i].crossover(self.Pc)
            self.populations[i].mutate(self.Pm)

            self.populations[i].calculate_fitness()

            print(self.populations[1].chromosomes[0].fitness)
            print(self.populations[1].chromosomes[1].fitness)

        return self.make_cluster_from_medoids()
