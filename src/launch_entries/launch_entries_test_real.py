import numpy as np

from src.cluster.cluster_launch_entry import ClusterLaunchEntry
from src.common.dataset import Dataset
from src.vrptw.vrptw_launch_entry import VRPTWLaunchEntry

dataset_arr = np.array([
    # Dataset(data_file='c104_mod.txt',
    #         dataset_type='C', dim=100),
    # Dataset(data_file='r110_mod.txt',
    #         dataset_type='R', dim=100),
    Dataset(data_file='rc103_mod.txt',
            dataset_type='RC', dim=100),

    # Dataset(data_file='C1_2_4_mod.TXT',
    #         dataset_type='C', dim=200),
    # Dataset(data_file='R1_2_10_mod.TXT',
    #         dataset_type='R', dim=200),
    # Dataset(data_file='RC1_2_3_mod.TXT',
    #         dataset_type='RC', dim=200),
    #
    # Dataset(data_file='C1_4_4_mod.TXT',
    #         dataset_type='C', dim=400),
    # Dataset(data_file='R1_4_10_mod.TXT',
    #         dataset_type='R', dim=400),
    # Dataset(data_file='RC1_4_3_mod.TXT',
    #         dataset_type='RC', dim=400),
])

vrptw_launch_entry = VRPTWLaunchEntry(vrptw_entry_id='test_real',
                                      k3_arr=np.array([2.0, 10.0, 100.0]),
                                      dataset_arr=dataset_arr,
                                      is_text=True,
                                      custom_cluster_launch_entry=ClusterLaunchEntry(P=25, ng=5),
                                      plot_stats_type_arr=np.array([
                                          'late_time_part', 'wait_time_bks'
                                      ]))
