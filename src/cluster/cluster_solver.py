import heapq
import math
import os
from multiprocessing import Pool

import numpy as np
from numpy import inf

from src.cluster.chromosome import Chromosome
from src.cluster.dm_chromosome import DMChromosome
from src.cluster.dm_population import DMPopulation
from src.cluster.population import Population
from src.common import fp_growth
from src.common.utils import make_cluster_from_medoids


class ClusterSolver:
    def __init__(self, distances, Z=10, P=20, ng=25, Pc=0.65, Pm=0.2, Pmb=0.05, k=None,
                 numpy_rand=None,
                 dm_size=4, dm_ng=5):
        self._distances = distances

        # Find k number based on number of customers K and
        # acceptable TSPTW subproblem size Z
        if k is None:
            K = len(distances)
            self._k = math.ceil(K / Z)
        else:
            self._k = k

        self._P = P
        self._ng = ng

        self._Pc = Pc
        self._Pm = Pm
        self._Pmb = Pmb

        self._numpy_rand = np.random.RandomState(42)

        self._dm_size = dm_size
        self._dm_ng = dm_ng
        self._dm_cur_priority_list = []
        self._dm_threshold = math.ceil(len(self._distances) * 0.03)
        self._dm_min_pattern_length = math.ceil(len(self._distances) * 0.03)

        self._DM_MAX_PATTERNS_SEARCH = 10
        self._DM_MAX_PRIORITY_LIST_SIZE = self._k

        self._BASE_DIR = os.path.abspath(os.curdir)

    # параллелизм здесь внутри будет для режима data_mining
    def solve_cluster_core_data_mining(self):
        # Заданное число итераций для dm режима
        global_best_res = None
        global_best_fitness = inf
        for i in range(self._dm_ng):
            print("DM iteration: {}. Best fitness: {}".format(i, global_best_fitness))

            patterns = {}
            patterns_search_i = 0

            cur_best_res = None
            cur_best_fitness = None
            while patterns_search_i < self._DM_MAX_PATTERNS_SEARCH and len(patterns) == 0:
                print("DM. dm: {}, patterns_search_i: {}.".format(i, patterns_search_i))

                # Получаем список с кластерами для каждого запуска
                res, cur_best_res, cur_best_fitness = self.make_multistart_genetic()
                # Помещаем все кластеры из разных потоков в один глобальный список
                flatten_res = np.zeros(dtype=int, shape=(self._dm_size * self._k, res[0][0].shape[0]))
                for p, sol in enumerate(res):
                    for q, cluster in enumerate(sol):
                        flatten_res[p * self._dm_size + q] = cluster

                # запуск fp-growth для поиска лучшей хромосомы из набора res
                patterns = fp_growth.find_frequent_patterns(flatten_res.tolist(), self._dm_threshold)
                filtered_patterns = {}
                for key in patterns:
                    if all(-1 != el and 0 != el for el in key):
                        filtered_patterns[key] = patterns[key]
                patterns = filtered_patterns

                patterns_search_i += 1

            if len(patterns) == 0:
                print("DM. Solution can't be found")
                return None

            for itemset in patterns.keys():
                freq = patterns[itemset]
                if len(itemset) >= self._dm_min_pattern_length \
                        and not any(itemset in item[1] for item in self._dm_cur_priority_list):
                    if len(self._dm_cur_priority_list) == self._DM_MAX_PRIORITY_LIST_SIZE:
                        heapq.heapreplace(self._dm_cur_priority_list, (freq, itemset))
                    else:
                        heapq.heappush(self._dm_cur_priority_list, (freq, itemset))

            if cur_best_fitness < global_best_fitness:
                global_best_res = cur_best_res
                global_best_fitness = cur_best_fitness

        return global_best_res

    def make_multistart_genetic(self):
        # Мультистарт
        rand_arr = self._numpy_rand.rand(self._dm_size, 1)
        np_rand_arr = [np.random.RandomState(int(100 * i)) for i in rand_arr]
        res = []

        with Pool(self._dm_size) as p:
            current_best_chromosome, population = self.init(
                DMPopulation(self._P, self._k, self._distances, self._dm_cur_priority_list),
                DMChromosome(self._k, self._distances, self._dm_cur_priority_list))
            args = [(population, current_best_chromosome, np_rand) for np_rand in np_rand_arr]
            result = p.starmap(self._solve, args)

            best_res = None
            cur_best_fitness = inf
            for chrom in result:
                clusters, _ = make_cluster_from_medoids(self._distances, self._dm_cur_priority_list, chrom.genes)
                res.append(clusters)

                if best_res is None:
                    best_res = clusters

                if chrom.fitness < cur_best_fitness:
                    cur_best_fitness = chrom.fitness
                    best_res = clusters

        return np.array(res), best_res, cur_best_fitness

    def solve(self):
        current_best_chromosome, population = self.init(Population(self._P, self._k, self._distances),
                                                        Chromosome(self._k, self._distances))
        res_chromosome = self._solve(population, current_best_chromosome, self._numpy_rand)
        clusters, _ = make_cluster_from_medoids(self._distances, self._dm_cur_priority_list, res_chromosome.genes)
        return clusters

    def _solve(self, population, cur_best_chromosome, np_rand):
        self.print_current_iteration_info(0, cur_best_chromosome, population)
        for i in range(1, self._ng):
            population = population.selection(np_rand)
            population.crossover(self._Pc, self._Pmb, np_rand)
            population.mutate(self._Pm, np_rand)

            population.calculate_fitness()
            cur_best_chromosome = self._get_new_best_chromosome(cur_best_chromosome, population)

            self.print_current_iteration_info(i, cur_best_chromosome, population)

        return cur_best_chromosome

    def init(self, population, chromosome):
        population.generate_random_population(self._numpy_rand)
        population.calculate_fitness()
        current_best_chromosome = self._get_new_best_chromosome(chromosome, population)
        return current_best_chromosome, population

    def print_current_iteration_info(self, i, current_best_chromosome, current_population):
        print("---------")
        print("Iteration: {}".format(i))
        print("Best chromosome fitness: {}".format(current_best_chromosome.fitness))
        # print("All chromosomes genes: {}".format(
        #     [chromosome.genes.tolist() for chromosome in current_population.chromosomes]))
        print("Best genes: {}".format(current_best_chromosome.genes))
        print("---------")

    def _get_new_best_chromosome(self, current_best_chromosome, population):
        best_chromosome_ind = population.find_best_chromosome_ind()
        best_chromosome = population.chromosomes[best_chromosome_ind]
        print("cur fitness: {}".format(best_chromosome.fitness))
        if best_chromosome.fitness < current_best_chromosome.fitness:
            return best_chromosome
        return current_best_chromosome
