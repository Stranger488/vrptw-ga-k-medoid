# --- test, r109_reduced --- #

import numpy as np


test_dataset = {
    'data_file': 'test.txt',
    'output_dir': 'test_output/',
    'plot': False,
    'name': 'test',
    'text': True
}

mapping = ['test_series']
dims_array = np.array([6, ])
k3_array = np.array([2.0, 10.0, 100.0, 1000.0])

testing_datasets = [[test_dataset, ], ]
