import os

import numpy as np

from src.cluster.cluster_launch_entry import ClusterLaunchEntry
from src.common.dataset import Dataset
from src.vrptw.vrptw_launch_entry import VRPTWLaunchEntry

dataset_arr = np.array([
    Dataset(data_file='c104_mod.txt',
            dataset_type='C', dim=100),
    Dataset(data_file='r110_mod.txt',
            dataset_type='R', dim=100),
    Dataset(data_file='rc103_mod.txt',
            dataset_type='RC', dim=100),

    Dataset(data_file='C1_2_4_mod.TXT',
            dataset_type='C', dim=200),
    Dataset(data_file='R1_2_10_mod.TXT',
            dataset_type='R', dim=200),
    Dataset(data_file='RC1_2_3_mod.TXT',
            dataset_type='RC', dim=200),

    Dataset(data_file='C1_4_4_mod.TXT',
            dataset_type='C', dim=400),
    Dataset(data_file='R1_4_10_mod.TXT',
            dataset_type='R', dim=400),
    Dataset(data_file='RC1_4_3_mod.TXT',
            dataset_type='RC', dim=400),

    Dataset(data_file='C1_6_4_mod.TXT',
            dataset_type='C', dim=600),
    Dataset(data_file='R1_6_10_mod.TXT',
            dataset_type='R', dim=600),
    Dataset(data_file='RC1_6_3_mod.TXT',
            dataset_type='RC', dim=600),

    # Dataset(data_file='C1_8_4_mod.TXT',
    #         dataset_type='C', dim=800),
    # Dataset(data_file='R1_8_10_mod.TXT',
    #         dataset_type='R', dim=800),
    # Dataset(data_file='RC1_8_3_mod.TXT',
    #         dataset_type='RC', dim=800),
    #
    # Dataset(data_file='C1_10_4_mod.TXT',
    #         dataset_type='C', dim=1000),
    # Dataset(data_file='R1_10_10_mod.TXT',
    #         dataset_type='R', dim=1000),
    # Dataset(data_file='RC1_10_3_mod.TXT',
    #         dataset_type='RC', dim=1000),
])

vrptw_launch_entry = VRPTWLaunchEntry(vrptw_entry_id='testing',
                                      k3_arr=np.array([2.0, 10.0, 100.0]),
                                      custom_cluster_launch_entry=ClusterLaunchEntry(P=100, ng_arr=[30, 20, 15, 15, 20],
                                                                                     dm_ng=5,
                                                                                     dm_size=os.cpu_count()
                                                                                     ),
                                      plot_stats_type_arr=np.array([
                                          'time_common_stats',

                                          'distance_stats',
                                          'distance_bks_stats',

                                          'avg_wait_time_stats',
                                          'avg_late_time_stats',
                                          'avg_wait_time_bks_stats',
                                          'avg_late_time_bks_stats',
                                      ])
                                      )
