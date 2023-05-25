from __future__ import annotations

from src.cluster.chromosome import Chromosome
from src.common.utils import make_cluster_from_medoids


class DMChromosome(Chromosome):
    def __init__(self, chromosome_size, distances, dm_priority_list):
        super().__init__(chromosome_size, distances)
        self.dm_priority_list = dm_priority_list

    def calculate_fitness(self):
        clusters, costs_sum = make_cluster_from_medoids(self._distances, self.dm_priority_list, self.genes)
        self.fitness = costs_sum

        return costs_sum
