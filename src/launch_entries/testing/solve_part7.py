import os

import numpy as np

from src.cluster.cluster_launch_entry import ClusterLaunchEntry
from src.common.dataset import Dataset
from src.vrptw.vrptw_launch_entry import VRPTWLaunchEntry

dataset_arr = np.array([
    Dataset(data_file='R1_10_10_mod.TXT',
            dataset_type='R', dim=1000),
])

vrptw_launch_entry = VRPTWLaunchEntry(vrptw_entry_id='testing_part7',
                                      k3_arr=np.array([2.0, 10.0, 100.0]),
                                      dataset_arr=dataset_arr,
                                      is_text=True,
                                      custom_cluster_launch_entry=ClusterLaunchEntry(P=100, ng_arr=[30, 20, 15, 15, 20],
                                                                                     dm_ng=5,
                                                                                     dm_size=os.cpu_count()
                                                                                     )
                                      )
