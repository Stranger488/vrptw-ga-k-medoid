# --- c1x4, r1x10, rc1x3 --- #
# 1000 customers

import numpy as np


C1_10_4_dataset = {
    'data_file': 'C1_10_4_mod.txt',
    'output_dir': 'C1_10_4_output/',
    'plot': False,
    'name': 'C1_10_4',
    'text': False
}

mapping = ['C', ]
dims_array = np.array([1000, ])
k3_array = np.array([2.0, 10.0, 100.0])

testing_datasets = [[C1_10_4_dataset, ], ]
