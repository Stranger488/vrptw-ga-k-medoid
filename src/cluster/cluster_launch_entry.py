import os

import numpy as np
from numpy.core.records import ndarray

from src.common.dataset import Dataset
from src.common.utils import get_common_entry_id


class ClusterLaunchEntry:
    def __init__(self, k1=1.0, k2=1.5, k3=2.0,
                 alpha1=0.5, alpha2=0.5,
                 Z=10, P=100, ng=25,
                 Pc=0.65, Pm=0.2, Pmb=0.05,
                 proc_count=os.cpu_count(),
                 dataset: Dataset = None):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.Z = Z
        self.P = P
        self.ng = ng
        self.Pc = Pc
        self.Pm = Pm
        self.Pmb = Pmb

        self.proc_count = proc_count
        self.dataset = dataset

        self.result = None
        self.common_id = get_common_entry_id(self.k3, self.dataset)

    @classmethod
    def get_all_launch_entries(cls, k3_arr, dataset_arr, proc_count,
                               cluster_launch_entry=None) -> ndarray:
        if cluster_launch_entry is not None:
            return np.array([cls(k1=cluster_launch_entry.k1, k2=cluster_launch_entry.k2,
                                 k3=k3,
                                 alpha1=cluster_launch_entry.alpha1, alpha2=cluster_launch_entry.alpha2,
                                 Z=cluster_launch_entry.Z, P=cluster_launch_entry.P,
                                 ng=cluster_launch_entry.ng, Pc=cluster_launch_entry.Pc,
                                 Pm=cluster_launch_entry.Pm, Pmb=cluster_launch_entry.Pmb,
                                 proc_count=proc_count, dataset=dataset)
                             for k3 in k3_arr for dataset in dataset_arr])

        return np.array([cls(k3=k3, proc_count=proc_count, dataset=dataset)
                         for k3 in k3_arr for dataset in dataset_arr])

