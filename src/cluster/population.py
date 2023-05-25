import numpy as np

from src.cluster.chromosome import Chromosome


class Population:
    def __init__(self, population_size, chromosome_size, distances):
        self._population_size = population_size
        self._distances = distances
        self.chromosomes = np.array(
            [Chromosome(chromosome_size, self._distances) for _ in range(self._population_size)],
            dtype=Chromosome)

    def generate_random_population(self, numpy_random):
        # Generate random genes for every chromosome
        np.fromiter(
            (c.generate_random_chromosome(numpy_random) for c in self.chromosomes),
            count=self._population_size,
            dtype=Chromosome
        )

    def calculate_fitness(self):
        # Calculate fitness for every chromosome in population
        np.fromiter(
            (c.calculate_fitness() for c in self.chromosomes),
            count=self._population_size,
            dtype=Chromosome
        )

    def selection(self, numpy_random):
        return self._roulette_selection(numpy_random)

    def crossover(self, crossover_prob, mut_prob, numpy_random):
        return self._dmx_crossover(crossover_prob, mut_prob, numpy_random)

    def mutate(self, prob, numpy_random):
        np.fromiter(
            (c.mutate(numpy_random) if prob > round(numpy_random.random(), 4) else None for c in self.chromosomes),
            count=self._population_size,
            dtype=Chromosome
        )

    def find_best_chromosome_ind(self):
        return np.argmin([c.fitness for c in self.chromosomes])

    def _dmx_crossover(self, crossover_prob, mut_prob, numpy_random):
        for i in range(0, self._population_size - 1, 2):
            if crossover_prob > round(numpy_random.random(), 4):
                mixed_gene = np.concatenate((self.chromosomes[i].genes, self.chromosomes[i + 1].genes))
                numpy_random.shuffle(mixed_gene)

                # TODO: стоит ли округлять numpy_random.random()?

                self._apply_builtin_mutation(mixed_gene, mut_prob, numpy_random)

                numpy_random.shuffle(mixed_gene)

                child1 = Chromosome(self.chromosomes[0].chromosome_size, self._distances)
                child2 = Chromosome(self.chromosomes[0].chromosome_size, self._distances)

                self._fill_children(child1, child2, mixed_gene)

                self.chromosomes[i] = child1
                self.chromosomes[i + 1] = child2

    def _fill_children(self, child1, child2, mixed_gene):
        k = 0  # index in child
        m = 0  # index in mixed_gene
        while k < child1.genes.size:
            if mixed_gene[m] not in child1.genes:
                child1.genes[k] = mixed_gene[m]
                k += 1
            m += 1
        k = 0  # index in child
        m = 0  # index in mixed_gene
        while k < child1.genes.size:
            if mixed_gene[mixed_gene.size - m - 1] not in child2.genes:
                child2.genes[k] = mixed_gene[mixed_gene.size - m - 1]
                k += 1
            m += 1

    def _apply_builtin_mutation(self, mixed_gene, mut_prob, numpy_random):
        applied = []
        for k in range(int(mixed_gene.size / 2)):
            if mut_prob > round(numpy_random.random(), 4):
                rand = numpy_random.randint(self._distances[0].size - 1)

                while rand in applied:
                    if rand < self._distances[0].size - 1:
                        rand += 1
                    else:
                        rand = 0

                applied.append(rand)
                mixed_gene[k] = rand

    def _roulette_selection(self, numpy_random):
        chrom_size = self.chromosomes[0].chromosome_size
        fitness = np.array([chromosome.fitness for chromosome in self.chromosomes])

        # использование np.sum() приводит к массиву вероятностей, который в сумме не дает 1.0, поэтому оставлен sum()
        probs = fitness / sum(fitness)

        new_population = Population(self._population_size, chrom_size, self._distances)
        new_population.chromosomes = numpy_random.choice(self.chromosomes, size=self._population_size, p=probs)

        return new_population
