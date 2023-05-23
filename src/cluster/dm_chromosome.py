from __future__ import annotations

import numpy as np

from src.cluster.chromosome import Chromosome


class DMChromosome(Chromosome):
    def __init__(self, chromosome_size, distances, dm_priority_list):
        super().__init__(chromosome_size, distances)
        self.dm_priority_list = dm_priority_list

    def calculate_fitness(self):
        costs_sum = 0.0
        points_size = int(np.ceil(self._distances[0].size / self.genes.size))
        approved = np.arange(self._distances[0].size)

        all_priority_list = np.array([])
        if len(self.dm_priority_list) != 0:
            # Пока что берется одна устоявшаяся группа
            all_priority_list = np.array(self.dm_priority_list[0][1])

        # Убрать медоиды из поиска
        approved = np.delete(approved, np.ravel([np.where(approved == med) for med in self.genes]))
        if all_priority_list.size != 0:
            priority_ = [np.where(approved == ind) for ind in all_priority_list]
            mod_priority = [el for el in priority_ if len(el[0]) != 0]
            approved = np.delete(approved, np.ravel(mod_priority))

        for gene in self.genes:
            for _ in range(points_size - 1):
                if all_priority_list.size != 0:
                    # Пока что всего один список - одна группа

                    # Строка с расстояниями до других вершин
                    cur_dist = self._distances[gene]

                    # Ищем значение и индекс ближайшей вершины
                    cur_min = cur_dist[all_priority_list].min()
                    cur_min_ind = np.ravel(np.where(cur_dist == cur_min))[0]

                    costs_sum += cur_min

                    # Удаляем из списка разрешенных
                    all_priority_list = all_priority_list[all_priority_list != cur_min_ind]

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
