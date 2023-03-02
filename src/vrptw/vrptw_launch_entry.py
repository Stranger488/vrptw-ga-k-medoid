import os

import numpy as np

from src.cluster.cluster_launch_entry import ClusterLaunchEntry
from src.tsptw.tsptw_launch_entry import TSPTWLaunchEntry


class VRPTWLaunchEntry:
    def __init__(self, vrptw_entry_id, c_D=1.0, c_T=1.0, c_L=1.5,
                 k3_arr=np.array([2.0, ]),
                 dataset_arr=None,
                 proc_count=os.cpu_count(),
                 is_text: bool = False,
                 plot_stats_type_arr=None,
                 custom_cluster_launch_entry: ClusterLaunchEntry = ClusterLaunchEntry(),
                 custom_tsptw_launch_entry: TSPTWLaunchEntry = TSPTWLaunchEntry()):
        self.cluster_launch_entry_arr = ClusterLaunchEntry.get_all_launch_entries(
            k3_arr=k3_arr,
            dataset_arr=dataset_arr,
            proc_count=proc_count,
            cluster_launch_entry=custom_cluster_launch_entry
        )
        self.tsptw_launch_entry_arr = TSPTWLaunchEntry.get_all_launch_entries(
            k3_arr=k3_arr,
            dataset_arr=dataset_arr,
            proc_count=proc_count,
            tsptw_launch_entry=custom_tsptw_launch_entry
        )

        self.c_D = c_D
        self.c_T = c_T
        self.c_L = c_L

        self.is_text = is_text
        self.proc_count = proc_count
        self.plot_stats_type_arr = plot_stats_type_arr

        self.vrptw_entry_id = vrptw_entry_id
        self.dataset_arr = dataset_arr

        self.BASE_DIR = os.path.abspath(os.curdir)
        self.CLUSTER_OUTPUT = self.BASE_DIR + '/vrptw_result/' + self.vrptw_entry_id + '/cluster_result/'
        self.TSPTW_OUTPUT = self.BASE_DIR + '/vrptw_result/' + self.vrptw_entry_id + '/tsptw_result/'
        self.EVALUATION_OUTPUT = self.BASE_DIR + '/vrptw_result/' + self.vrptw_entry_id + '/evaluation/'
        self.PLOT_STATS_OUTPUT = self.BASE_DIR + '/vrptw_result/' + self.vrptw_entry_id + '/plot_stats/'
