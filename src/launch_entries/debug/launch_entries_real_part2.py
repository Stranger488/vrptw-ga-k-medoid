import numpy as np

from src.cluster.cluster_launch_entry import ClusterLaunchEntry
from src.common.dataset import Dataset
from src.vrptw.vrptw_launch_entry import VRPTWLaunchEntry

dataset_arr = np.array([
    Dataset(data_file='C1_6_4_mod.TXT',
            dataset_type='C', dim=600),
    Dataset(data_file='R1_6_10_mod.TXT',
            dataset_type='R', dim=600),
    Dataset(data_file='RC1_6_3_mod.TXT',
            dataset_type='RC', dim=600),

    Dataset(data_file='C1_8_4_mod.TXT',
            dataset_type='C', dim=800),
    Dataset(data_file='R1_8_10_mod.TXT',
            dataset_type='R', dim=800),
    Dataset(data_file='RC1_8_3_mod.TXT',
            dataset_type='RC', dim=800),

    Dataset(data_file='C1_10_4_mod.TXT',
            dataset_type='C', dim=1000),
    Dataset(data_file='R1_10_10_mod.TXT',
            dataset_type='R', dim=1000),
    Dataset(data_file='RC1_10_3_mod.TXT',
            dataset_type='RC', dim=1000),
])

vrptw_launch_entry = VRPTWLaunchEntry(vrptw_entry_id='real_part2',
                                      k3_arr=np.array([2.0, 10.0, 100.0]),
                                      dataset_arr=dataset_arr,
                                      is_text=True,
                                      custom_cluster_launch_entry=ClusterLaunchEntry(P=100, ng=100,
                                                                                     dm_ng=10)
                                      )
