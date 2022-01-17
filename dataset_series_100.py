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
r110_dataset = {
    'data_file': 'r110_mod.txt',
    'output_dir': 'r110_output/',
    'plot': False,
    'name': 'r110',
    'text': False
}
rc103_dataset = {
    'data_file': 'rc103_mod.txt',
    'output_dir': 'rc103_output/',
    'plot': False,
    'name': 'rc103',
    'text': False
}

mapping = ['C', 'R', 'RC']
dims_array = np.array([100, ])
k3_array = np.array([2.0, ])

testing_datasets = [[c104_dataset, ],
                    [r110_dataset, ],
                    [rc103_dataset, ]]
