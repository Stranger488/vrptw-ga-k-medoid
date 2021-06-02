# --- c1x4, r1x10, rc1x3 --- #
# from 100 customers to 600

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

C1_2_4_dataset = {
    'data_file': 'C1_2_4_mod.txt',
    'output_dir': 'C1_2_4_output/',
    'plot': False,
    'name': 'C1_2_4',
    'text': False
}
R1_2_10_dataset = {
    'data_file': 'R1_2_10_mod.txt',
    'output_dir': 'R1_2_10_output/',
    'plot': False,
    'name': 'R1_2_10',
    'text': False
}
RC1_2_3_dataset = {
    'data_file': 'RC1_2_3_mod.txt',
    'output_dir': 'RC1_2_3_output/',
    'plot': False,
    'name': 'RC1_2_3',
    'text': False
}

mapping = ['C', 'R', 'RC']
dims_array = np.array([100, 200, ])
k3_array = np.array([2.0, 10.0])

testing_datasets = [[c104_dataset, C1_2_4_dataset],
                    [r110_dataset, R1_2_10_dataset],
                    [rc103_dataset, RC1_2_3_dataset]]
