import math

import numpy as np


class Chromosome:
    def __init__(self, chromosome_size, distances):
        self.chromosome_size = chromosome_size
        self.genes = np.empty(self.chromosome_size, dtype=int)
        self.fitness = -math.inf

        self.distances = distances

    def generate_random_chromosome(self):
        self.genes = np.random.choice(np.arange(0, self.distances[0].size), replace=False, size=self.chromosome_size)

    def calculate_fitness(self):
        costs_sum = 0.0

        points_size = int(np.ceil(self.distances[0].size / self.genes.size))

        tmp = np.array([[-1 for _ in range(points_size)] for _ in range(self.genes.size)])

        for i, gene in enumerate(self.genes):
            for j in range(points_size):
                if j == 0:
                    tmp[i][0] = gene
                    costs_sum += self.distances[gene][0]
                    continue

                cur_min_ind = -1
                cur_min = math.inf
                for k, el in enumerate(self.distances[gene]):
                    if gene != k and el < cur_min and k not in tmp:
                        cur_min = el
                        cur_min_ind = k
                tmp[i][j] = cur_min_ind
                costs_sum += self.distances[gene][cur_min_ind]

        self.fitness = costs_sum

        return costs_sum

    def mutate(self):
        rand_mutate_ind = np.random.randint(self.distances[0].size - self.chromosome_size)

        if rand_mutate_ind in self.genes:
            rand_mutate_ind += 1

        rand_gen_pos = np.random.randint(self.genes.size)
        self.genes[rand_gen_pos] = rand_mutate_ind
