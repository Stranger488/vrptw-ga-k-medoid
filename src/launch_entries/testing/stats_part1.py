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
            dataset_type='RC', dim=200)
])

vrptw_launch_entry = VRPTWLaunchEntry(vrptw_entry_id='testing_part1',
                                      k3_arr=np.array([2.0, 10.0, 100.0]),
                                      dataset_arr=dataset_arr,
                                      custom_cluster_launch_entry=ClusterLaunchEntry(P=50, ng_arr=[30, 20, 15, 15, 20],
                                                                                     dm_ng=5,
                                                                                     dm_size=4
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