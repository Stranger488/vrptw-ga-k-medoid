from __future__ import annotations

import math

import numpy as np

from src.common.utils import make_cluster_from_medoids


class Chromosome:
    def __init__(self, chromosome_size, distances):
        self.chromosome_size = chromosome_size
        self.genes = np.full(self.chromosome_size, -1, dtype=int)

        self.fitness = math.inf

        self._distances = distances

    def generate_random_chromosome(self, numpy_random):
        self.genes = numpy_random.choice(self._distances[0].size, replace=False, size=self.chromosome_size)

    def calculate_fitness(self):
        clusters, costs_sum = make_cluster_from_medoids(self._distances, [], self.genes)
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
