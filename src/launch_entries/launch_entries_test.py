import numpy as np

from src.cluster.cluster_launch_entry import ClusterLaunchEntry
from src.common.dataset import Dataset
from src.vrptw.vrptw_launch_entry import VRPTWLaunchEntry

dataset_arr = np.array([
    Dataset(data_file='test_C.txt',
            dataset_type='C', dim=6),
    Dataset(data_file='test_R.txt',
            dataset_type='R', dim=6),
    Dataset(data_file='test_RC.txt',
            dataset_type='RC', dim=6),
])

vrptw_launch_entry = VRPTWLaunchEntry(vrptw_entry_id='test',
                                      k3_arr=np.array([2.0, 10.0, 100.0]),
                                      dataset_arr=dataset_arr,
                                      is_text=True,
                                      custom_cluster_launch_entry=ClusterLaunchEntry(P=25, ng=5),
                                      plot_stats_type_arr=np.array([
                                          'std_wait_time_bks_stats',
                                      ]))
