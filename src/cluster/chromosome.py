from __future__ import annotations

import math

import numpy as np


class Chromosome:
    def __init__(self, chromosome_size, distances):
        self.chromosome_size = chromosome_size
        self.genes = np.full(self.chromosome_size, -1, dtype=int)

        self.fitness = math.inf

        self._distances = distances

    def generate_random_chromosome(self, numpy_random):
        self.genes = numpy_random.choice(self._distances[0].size, replace=False, size=self.chromosome_size)

    def calculate_fitness(self):
        costs_sum = 0.0

        points_size = int(np.ceil(self._distances[0].size / self.genes.size))

        approved = np.arange(self._distances[0].size)

        # Убрать медоиды из поиска
        approved = np.delete(approved, np.ravel([np.where(approved == med) for med in self.genes]))

        for gene in self.genes:
            for _ in range(points_size - 1):
                if approved.size != 0:
                    # Строка с расстояниями до других вершин
                    cur_dist = self._distances[gene]

                    # Ищем значение и индекс ближайшей вершины
                    cur_min = cur_dist[approved].min()
                    cur_min_ind = np.ravel(np.where(cur_dist == cur_min))[0]

                    costs_sum += cur_min

                    # Удаляем из списка разрешенных
                    approved = approved[approved != cur_min_ind]

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
