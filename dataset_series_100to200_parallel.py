# --- c1x4, r1x10, rc1x3 --- #
# from 100 customers to 200


c104_dataset = {
    'data_file': 'c104_mod.txt',
    'output_dir': 'c104_output_2/',
    'plot': False,
    'name': 'c104',
    'text': False,
    'mapping': 'C',
    'dim': 100,
    'k3': 2.0
}
r110_dataset = {
    'data_file': 'r110_mod.txt',
    'output_dir': 'r110_output_2/',
    'plot': False,
    'name': 'r110',
    'text': False,
    'mapping': 'C',
    'dim': 100,
    'k3': 2.0
}
rc103_dataset = {
    'data_file': 'rc103_mod.txt',
    'output_dir': 'rc103_output_2/',
    'plot': False,
    'name': 'rc103',
    'text': False,
    'mapping': 'C',
    'dim': 100,
    'k3': 2.0
}

C1_2_4_dataset = {
    'data_file': 'C1_2_4_mod.txt',
    'output_dir': 'C1_2_4_output_2/',
    'plot': False,
    'name': 'C1_2_4',
    'text': False,
    'mapping': 'C',
    'dim': 200,
    'k3': 2.0
}
R1_2_10_dataset = {
    'data_file': 'R1_2_10_mod.txt',
    'output_dir': 'R1_2_10_output_2/',
    'plot': False,
    'name': 'R1_2_10',
    'text': False,
    'mapping': 'C',
    'dim': 200,
    'k3': 2.0
}
RC1_2_3_dataset = {
    'data_file': 'RC1_2_3_mod.txt',
    'output_dir': 'RC1_2_3_output_2/',
    'plot': False,
    'name': 'RC1_2_3',
    'text': False,
    'mapping': 'R',
    'dim': 200,
    'k3': 2.0
}

testing_datasets = [c104_dataset, C1_2_4_dataset,
                    r110_dataset, R1_2_10_dataset,
                    rc103_dataset, RC1_2_3_dataset, ]
