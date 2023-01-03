import os

import numpy as np
from numpy.core.records import ndarray

from src.common.dataset import Dataset
from src.common.utils import get_common_entry_id


class TSPTWLaunchEntry:
    def __init__(self, k1=10, k2=100, route='closed',
                 penalty_value=1000, population_size=50,
                 mutation_rate=0.1, elite=1, generations=50,
                 proc_count=os.cpu_count(),
                 dataset: Dataset = None,
                 cluster_k3=None):
        self.k1 = k1
        self.k2 = k2
        self.route = route
        self.penalty_value = penalty_value
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite = elite
        self.generations = generations

        self.proc_count = proc_count
        self.dataset = dataset

        self.cluster_k3 = cluster_k3
        self.common_id = get_common_entry_id(self.cluster_k3, self.dataset)

    @classmethod
    def get_all_launch_entries(cls, k3_arr, dataset_arr, proc_count,
                               tsptw_launch_entry=None) -> ndarray:
        if tsptw_launch_entry is not None:
            return np.array([cls(k1=tsptw_launch_entry.k1, k2=tsptw_launch_entry.k2,
                                 route=tsptw_launch_entry.route, penalty_value=tsptw_launch_entry.penalty_value,
                                 population_size=tsptw_launch_entry.population_size,
                                 mutation_rate=tsptw_launch_entry.mutation_rate,
                                 elite=tsptw_launch_entry.elite, generations=tsptw_launch_entry.generations,
                                 proc_count=proc_count, dataset=dataset,
                                 cluster_k3=k3)
                             for k3 in k3_arr for dataset in dataset_arr])

        return np.array([cls(proc_count=proc_count, dataset=dataset, cluster_k3=k3)
                         for k3 in k3_arr for dataset in dataset_arr])
