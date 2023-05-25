import numpy as np

from src.cluster.dm_chromosome import DMChromosome
from src.cluster.population import Population


class DMPopulation(Population):
    def __init__(self, population_size, chromosome_size, distances, dm_priority_list):
        super().__init__(population_size, chromosome_size, distances)
        self.dm_priority_list = dm_priority_list
        self.chromosomes = np.array(
            [DMChromosome(chromosome_size, self._distances, self.dm_priority_list) for _ in
             range(self._population_size)],
            dtype=DMChromosome)

    def _dmx_crossover(self, crossover_prob, mut_prob, numpy_random):
        for i in range(0, self._population_size - 1, 2):
            if crossover_prob > round(numpy_random.random(), 4):
                mixed_gene = np.concatenate((self.chromosomes[i].genes, self.chromosomes[i + 1].genes))
                numpy_random.shuffle(mixed_gene)

                # TODO: стоит ли округлять numpy_random.random()?

                self._apply_builtin_mutation(mixed_gene, mut_prob, numpy_random)
                numpy_random.shuffle(mixed_gene)

                child1 = DMChromosome(self.chromosomes[0].chromosome_size, self._distances, self.dm_priority_list)
                child2 = DMChromosome(self.chromosomes[0].chromosome_size, self._distances, self.dm_priority_list)

                self._fill_children(child1, child2, mixed_gene)

                self.chromosomes[i] = child1
                self.chromosomes[i + 1] = child2

    # TODO: создание популяции и хромосомы в абстрактный метод вынести, чтобы не дублировать методы родителя
    def _roulette_selection(self, numpy_random):
        chrom_size = self.chromosomes[0].chromosome_size
        fitness = np.array([chromosome.fitness for chromosome in self.chromosomes])

        # использование np.sum() приводит к массиву вероятностей, который в сумме не дает 1.0, поэтому оставлен sum()
        probs = fitness / sum(fitness)

        new_population = DMPopulation(self._population_size, chrom_size, self._distances, self.dm_priority_list)
        new_population.chromosomes = numpy_random.choice(self.chromosomes, size=self._population_size, p=probs)

        return new_population
