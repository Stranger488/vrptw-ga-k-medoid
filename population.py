import math

import numpy as np

from chromosome import Chromosome


class Population:
    def __init__(self, population_size, chromosome_size, distances):
        self.population_size = population_size

        self.distances = distances

        self.chromosomes = np.array([])
        for _ in range(self.population_size):
            self.chromosomes = np.append(self.chromosomes, Chromosome(chromosome_size, self.distances))

    def generate_random_population(self, numpy_random):
        # Generate random genes for every chromosome
        for chromosome in self.chromosomes:
            chromosome.generate_random_chromosome(numpy_random)

    def calculate_fitness(self):
        # Calculate fitness for every chromosome in population
        for chromosome in self.chromosomes:
            chromosome.calculate_fitness()

    def selection(self, numpy_random):
        return self.roulette_selection(numpy_random)

    def roulette_selection(self, numpy_random):
        chrom_size = self.chromosomes[0].chromosome_size
        fitnesses = np.array([chromosome.fitness for chromosome in self.chromosomes])
        fitnesses_sum = np.sum(fitnesses)

        rel_fitnesses = np.array([fitness / fitnesses_sum for fitness in fitnesses])

        probs = np.array([np.sum(rel_fitnesses[:i + 1]) for i in range(len(rel_fitnesses))])

        new_population = Population(self.population_size, chrom_size, self.distances)
        new_population.generate_random_population(numpy_random)

        for i in range(chrom_size):
            rand = round(numpy_random.random(), 4)
            for j, chromosome in enumerate(self.chromosomes):
                if probs[j] > rand:
                    new_population.chromosomes[j] = chromosome
                    break

        new_population.calculate_fitness()

        return new_population

    def mutate(self, prob, numpy_random):
        for chromosome in self.chromosomes:
            if prob > round(numpy_random.random(), 4):
                chromosome.mutate(numpy_random)

    def find_best_chromosome(self):
        min_fitness = math.inf
        min_ind = -1
        for i, chromosome in enumerate(self.chromosomes):
            if chromosome.fitness < min_fitness:
                min_fitness = chromosome.fitness
                min_ind = i

        return self.chromosomes[min_ind]

    def dmx_crossover(self, crossover_prob, mut_prob, numpy_random):
        for i in range(0, self.population_size - 1, 2):
            if crossover_prob > round(numpy_random.random(), 4):
                mixed_gene = np.concatenate((self.chromosomes[i].genes, self.chromosomes[i + 1].genes))
                numpy_random.shuffle(mixed_gene)

                # Apply built-in mutation
                applied = []
                for k in range(mixed_gene.size):
                    if mut_prob > round(numpy_random.random(), 4):
                        rand = numpy_random.randint(0, self.distances[0].size - 1)

                        while rand in applied:
                            if rand < self.distances[0].size - 1:
                                rand += 1
                            else:
                                rand = 0

                        applied.append(rand)
                        mixed_gene[k] = rand

                numpy_random.shuffle(mixed_gene)

                child1 = Chromosome(self.chromosomes[0].chromosome_size, self.distances)
                child2 = Chromosome(self.chromosomes[0].chromosome_size, self.distances)

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

                self.chromosomes[i] = child1
                self.chromosomes[i + 1] = child2
