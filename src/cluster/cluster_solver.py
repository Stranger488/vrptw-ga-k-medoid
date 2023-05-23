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


class ClusterSolver:
    def __init__(self, distances, Z=10, P=20, ng=25, Pc=0.65, Pm=0.2, Pmb=0.05, k=None, numpy_rand=None):
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

        self._current_population = Population(self._P, self._k, self._distances)
        self._best_chromosome = Chromosome(self._k, self._distances)

        self._dm_size = 4
        self._dm_ng = 5
        self._dm_cur_priority_list = []
        self._dm_threshold = 10
        self._dm_min_pattern_length = 3

        self._DM_MAX_PATTERNS_SEARCH = 10
        self._DM_MAX_PRIORITY_LIST_SIZE = 10

        self._BASE_DIR = os.path.abspath(os.curdir)

    # параллелизм здесь внутри будет для режима data_mining
    def solve_cluster_core_data_mining(self):
        # Заданное число итераций для dm режима
        best_res = None
        for i in range(self._dm_ng):
            print("DM iteration: {}".format(i))

            patterns = {}
            patterns_search_i = 0
            while patterns_search_i < self._DM_MAX_PATTERNS_SEARCH and len(patterns) == 0:
                # Получаем список с кластерами для каждого запуска
                res, best_res = self.make_multistart_genetic()
                # Помещаем все кластеры из разных потоков в один глобальный список
                flatten_res = np.zeros(dtype=int, shape=(self._dm_size * self._k, res[0][0].shape[0]))
                for p, sol in enumerate(res):
                    for q, cluster in enumerate(sol):
                        flatten_res[p * self._dm_size + q] = cluster

                # запуск fp-growth для поиска лучшей хромосомы из набора res
                patterns = fp_growth.find_frequent_patterns(flatten_res.tolist(), self._dm_threshold)

                patterns_search_i += 1

            if len(patterns) == 0:
                print("DM. Solution can't be found")
                return np.empty((0, 0))

            for itemset in patterns.keys():
                freq = patterns[itemset]
                if len(itemset) >= self._dm_min_pattern_length \
                        and not any(itemset in item[1] for item in self._dm_cur_priority_list):
                    if len(self._dm_cur_priority_list) == self._DM_MAX_PRIORITY_LIST_SIZE:
                        heapq.heapreplace(self._dm_cur_priority_list, (freq, itemset))
                    else:
                        heapq.heappush(self._dm_cur_priority_list, (freq, itemset))

        return best_res

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
                clusters = self._make_cluster_from_medoids(chrom)
                res.append(clusters)

                if best_res is None:
                    best_res = clusters

                if chrom.fitness < cur_best_fitness:
                    cur_best_fitness = chrom.fitness
                    best_res = clusters

        return np.array(res), best_res

    def solve(self):
        current_best_chromosome, population = self.init(Population(self._P, self._k, self._distances),
                                                        Chromosome(self._k, self._distances))
        res_chromosome = self._solve(population, current_best_chromosome, self._numpy_rand)
        return self._make_cluster_from_medoids(res_chromosome)

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
        print("All chromosomes genes: {}".format(
            [chromosome.genes.tolist() for chromosome in current_population.chromosomes]))
        print("Best genes: {}".format(current_best_chromosome.genes))
        print("---------")

    def _get_new_best_chromosome(self, current_best_chromosome, population):
        best_chromosome_ind = population.find_best_chromosome_ind()
        best_chromosome = population.chromosomes[best_chromosome_ind]
        print("cur fitness: {}".format(best_chromosome.fitness))
        if best_chromosome.fitness < current_best_chromosome.fitness:
            return best_chromosome
        return current_best_chromosome

    def _make_cluster_from_medoids(self, cur_best_chromosome):
        medoids = cur_best_chromosome.genes
        points_size = int(np.ceil(self._distances[0].size / medoids.size))
        result = np.full((medoids.size, points_size), -1)

        approved = np.arange(self._distances[0].size)

        all_priority_list = np.array([])
        if len(self._dm_cur_priority_list) != 0:
            all_priority_list = np.array(self._dm_cur_priority_list[0][1])

        # Убрать медоиды из поиска, вручную их помещаем в кластер на нулевую позицию
        approved = np.delete(approved, np.ravel([np.where(approved == med) for med in medoids]))
        if all_priority_list.size != 0:
            priority_ = [np.where(approved == ind) for ind in all_priority_list]
            mod_priority = [el for el in priority_ if len(el[0]) != 0]
            approved = np.delete(approved, np.ravel(mod_priority))

        for i, gene in enumerate(medoids):
            for j in range(1, points_size):
                if all_priority_list.size != 0:
                    # Пока что всего один список - одна группа

                    # Строка с расстояниями до других вершин
                    cur_dist = self._distances[gene]

                    # Ищем значение и индекс ближайшей вершины
                    cur_min_ind = -1
                    cur_min = np.inf
                    for ind, el in enumerate(cur_dist):
                        if ind in approved and el < cur_min:
                            cur_min = el
                            cur_min_ind = ind

                    result[i][j] = cur_min_ind

                    # Удаляем из списка разрешенных
                    all_priority_list = all_priority_list[all_priority_list != cur_min_ind]

        for i, gene in enumerate(medoids):
            # Сразу помещаем медоид
            result[i][0] = gene
            for j in range(1, points_size):
                if approved.size != 0:
                    # Строка с расстояниями до других вершин
                    cur_dist = self._distances[gene]

                    # Ищем индекс ближайшей вершины
                    cur_min_ind = -1
                    cur_min = np.inf
                    for ind, el in enumerate(cur_dist):
                        if ind in approved and el < cur_min:
                            cur_min = el
                            cur_min_ind = ind

                    # cur_min_ind = np.ravel(np.where(cur_dist == cur_dist[approved].min()))[0]

                    result[i][j] = cur_min_ind

                    # Удаляем из списка разрешенных
                    approved = approved[approved != cur_min_ind]

        return result
