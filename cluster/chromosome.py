import math

import numpy as np


class Chromosome:
    def __init__(self, chromosome_size, distances):
        self.chromosome_size = chromosome_size
        self.genes = np.empty(self.chromosome_size, dtype=int)
        for i in range(self.genes.size):
            self.genes[i] = -1

        self.fitness = math.inf

        self._distances = distances

    def generate_random_chromosome(self, numpy_random):
        self.genes = numpy_random.choice(np.arange(0, self._distances[0].size), replace=False, size=self.chromosome_size)

    def calculate_fitness(self):
        costs_sum = 0.0

        points_size = int(np.ceil(self._distances[0].size / self.genes.size))
        deprecated = np.copy(self.genes)

        for i, gene in enumerate(self.genes):
            for j in range(points_size):
                if j == 0:
                    costs_sum += self._distances[gene][0]
                    continue

                cur_min_ind = -1
                cur_min = math.inf
                for k, el in enumerate(self._distances[gene]):
                    if gene != k and el < cur_min and k not in deprecated:
                        cur_min = el
                        cur_min_ind = k
                deprecated = np.append(deprecated, cur_min_ind)
                costs_sum += self._distances[gene][cur_min_ind]

        self.fitness = costs_sum

        return costs_sum

    def mutate(self, numpy_random):
        rand_mutate_ind = numpy_random.randint(self._distances[0].size)

        while rand_mutate_ind in self.genes:
            if rand_mutate_ind < self._distances[0].size - 1:
                rand_mutate_ind += 1
            else:
                rand_mutate_ind = 0

        rand_gen_pos = numpy_random.randint(self.genes.size)
        self.genes[rand_gen_pos] = rand_mutate_ind
