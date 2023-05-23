import numpy as np

from src.cluster.dm_chromosome import DMChromosome
from src.cluster.population import Population


class DMPopulation(Population):
    def __init__(self, population_size, chromosome_size, distances, dm_priority_list):
        super().__init__(population_size, chromosome_size, distances)
        self.chromosomes = np.array(
            [DMChromosome(chromosome_size, self._distances, dm_priority_list) for _ in range(self._population_size)],
            dtype=DMChromosome)
