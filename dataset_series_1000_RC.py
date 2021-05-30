# --- c1x4, r1x10, rc1x3 --- #
# 1000 customers

import numpy as np


RC1_10_3_dataset = {
    'data_file': 'RC1_10_3_mod.txt',
    'output_dir': 'RC1_10_3_output/',
    'plot': False,
    'name': 'RC1_10_3',
    'text': False
}

mapping = ['RC', ]
dims_array = np.array([1000, ])
k3_array = np.array([2.0, 10.0, 100.0])

testing_datasets = [[RC1_10_3_dataset, ], ]
