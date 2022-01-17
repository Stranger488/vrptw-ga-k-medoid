# --- c1x4 --- #
# 100 customers

import numpy as np

c104_dataset = {
    'data_file': 'c104_mod.txt',
    'output_dir': 'c104_output/',
    'plot': False,
    'name': 'c104',
    'text': False
}
C1_2_4_dataset = {
    'data_file': 'C1_2_4_mod.txt',
    'output_dir': 'C1_2_4_output/',
    'plot': False,
    'name': 'C1_2_4',
    'text': False
}
C1_4_4_dataset = {
    'data_file': 'C1_4_4_mod.txt',
    'output_dir': 'C1_4_4_output/',
    'plot': False,
    'name': 'C1_4_4',
    'text': False
}
C1_6_4_dataset = {
    'data_file': 'C1_6_4_mod.txt',
    'output_dir': 'C1_6_4_output/',
    'plot': False,
    'name': 'C1_6_4',
    'text': False
}

mapping = ['C', ]
dims_array = np.array([100, 200, 400, 600])
k3_array = np.array([2.0, ])

testing_datasets = [[c104_dataset, C1_2_4_dataset, C1_4_4_dataset, C1_6_4_dataset], ]
