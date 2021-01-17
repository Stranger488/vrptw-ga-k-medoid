import random

import numpy as np

from chromosome import Chromosome


class Population:
    def __init__(self, population_size, chromosome_size, distances):
        self.population_size = population_size

        self.distances = distances

        self.best_chromosome_ind = -1

        self.chromosomes = np.empty(self.population_size, dtype=Chromosome)
        self.chromosomes.fill(Chromosome(chromosome_size, self.distances))

    def generate_random_population(self):
        for chromosome in self.chromosomes:
            chromosome.generate_random_chromosome()

    def calculate_fitness(self):
        for chromosome in self.chromosomes:
            chromosome.calculate_fitness()

    def selection(self):
        return self.roulette_selection()

    def roulette_selection(self):
        chrom_size = self.chromosomes[0].chromosome_size
        fitnesses = np.array([chromosome.fitness for chromosome in self.chromosomes])
        fitnesses_sum = np.sum(fitnesses)

        rel_fitnesses = np.array([fitness / fitnesses_sum for fitness in fitnesses])

        probs = np.array([np.sum(rel_fitnesses[:i + 1]) for i in range(len(rel_fitnesses))])

        new_population = Population(self.population_size, chrom_size, self.distances)
        new_population.generate_random_population()

        for i in range(chrom_size):
            rand = random.random()
            for j, chromosome in enumerate(self.chromosomes):
                if probs[j] > rand:
                    new_population.chromosomes[j] = chromosome
                    break

        new_population.calculate_fitness()

        return new_population

    def crossover(self, prob):
        if prob > random.random():
            rand_parents_ind = np.random.choice(np.arange(0, self.population_size), replace=False, size=2)

            cut_point_ind = random.randint(1, self.chromosomes[0].chromosome_size - 1)

            parent1 = self.chromosomes[rand_parents_ind[0]]
            parent2 = self.chromosomes[rand_parents_ind[1]]

            child1 = Chromosome(self.chromosomes[0].chromosome_size, self.distances)
            child2 = Chromosome(self.chromosomes[0].chromosome_size, self.distances)

            for _ in range(self.chromosomes[0].chromosome_size):
                child1.genes[:cut_point_ind] = parent1.genes[:cut_point_ind]
                child1.genes[cut_point_ind:] = parent2.genes[cut_point_ind:]

                child2.genes[:cut_point_ind] = parent2.genes[:cut_point_ind]
                child2.genes[cut_point_ind:] = parent1.genes[cut_point_ind:]

            self.chromosomes[rand_parents_ind[0]] = child1
            self.chromosomes[rand_parents_ind[1]] = child2

    def mutate(self, prob):
        for chromosome in self.chromosomes:
            if prob > random.random():
                chromosome.mutate()

    def find_best_chrom(self):
        self.best_chromosome_ind = np.argmax(np.array([el.fitness for el in self.chromosomes]))