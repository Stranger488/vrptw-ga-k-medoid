# --- c1x4, r1x10, rc1x3 --- #
# 800 customers

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

C1_4_4_dataset = {
    'data_file': 'C1_4_4_mod.txt',
    'output_dir': 'C1_4_4_output/',
    'plot': False,
    'name': 'C1_4_4',
    'text': False
}
R1_4_10_dataset = {
    'data_file': 'R1_4_10_mod.txt',
    'output_dir': 'R1_4_10_output/',
    'plot': False,
    'name': 'R1_4_10',
    'text': False
}
RC1_4_3_dataset = {
    'data_file': 'RC1_4_3_mod.txt',
    'output_dir': 'RC1_4_3_output/',
    'plot': False,
    'name': 'RC1_4_3',
    'text': False
}

C1_6_4_dataset = {
    'data_file': 'C1_6_4_mod.txt',
    'output_dir': 'C1_6_4_output/',
    'plot': False,
    'name': 'C1_6_4',
    'text': False
}
R1_6_10_dataset = {
    'data_file': 'R1_6_10_mod.txt',
    'output_dir': 'R1_6_10_output/',
    'plot': False,
    'name': 'R1_6_10',
    'text': False
}
RC1_6_3_dataset = {
    'data_file': 'RC1_6_3_mod.txt',
    'output_dir': 'RC1_6_3_output/',
    'plot': False,
    'name': 'RC1_6_3',
    'text': False
}

C1_8_4_dataset = {
    'data_file': 'C1_8_4_mod.txt',
    'output_dir': 'C1_8_4_output/',
    'plot': False,
    'name': 'C1_8_4',
    'text': False
}
R1_8_10_dataset = {
    'data_file': 'R1_8_10_mod.txt',
    'output_dir': 'R1_8_10_output/',
    'plot': False,
    'name': 'R1_8_10',
    'text': False
}
RC1_8_3_dataset = {
    'data_file': 'RC1_8_3_mod.txt',
    'output_dir': 'RC1_8_3_output/',
    'plot': False,
    'name': 'RC1_8_3',
    'text': False
}

mapping = ['C', 'R', 'RC', ]
dims_array = np.array([100, 200, 400, 600, 800, ])
k3_array = np.array([2.0, 10.0, 100.0])

testing_datasets = [[c104_dataset, C1_2_4_dataset, C1_4_4_dataset, C1_6_4_dataset, C1_8_4_dataset, ],
                    [r110_dataset, R1_2_10_dataset, R1_4_10_dataset, R1_6_10_dataset, R1_8_10_dataset, ],
                    [rc103_dataset, RC1_2_3_dataset, RC1_4_3_dataset, RC1_6_3_dataset, RC1_8_3_dataset, ], ]
