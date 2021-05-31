# --- c1x4, r1x10, rc1x3 --- #
# 1000 customers

import numpy as np


R1_10_10_dataset = {
    'data_file': 'R1_10_10_mod.txt',
    'output_dir': 'R1_10_10_output/',
    'plot': False,
    'name': 'R1_10_10',
    'text': False
}

mapping = ['R', ]
dims_array = np.array([1000, ])
k3_array = np.array([2.0, 10.0, 100.0])

testing_datasets = [[R1_10_10_dataset, ], ]
