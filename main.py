import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kernel import Kernel
from statistics import Statistics
from plot import Plot

from mpl_toolkits.mplot3d.axis3d import Axis


# fix wrong z-offsets in 3d plot
def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0, 0, 1.0 / 4]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs


if not hasattr(Axis, "_get_coord_info_old"):
    Axis._get_coord_info_old = Axis._get_coord_info
Axis._get_coord_info = _get_coord_info_new

if __name__ == '__main__':
    plotter = Plot()
    kernel = Kernel()


    test_dataset = {
        'data_file': 'test.txt',
        'output_dir': 'test_output/',
        'plot': True,
        'name': 'test',
        'text': True,
        'method': 'cluster'
    }

    # kernel.solve_and_plot([test_dataset, ])


    test_dataset1 = {
        'data_file': 'test.txt',
        'output_dir': 'test1_output/',
        'plot': False,
        'name': 'test1',
        'text': True,
        'method': 'cluster'
    }
    test_dataset2 = {
        'data_file': 'test.txt',
        'output_dir': 'test2_output/',
        'plot': False,
        'name': 'test2',
        'text': True,
        'method': 'cluster'
    }
    test_dataset3 = {
        'data_file': 'test.txt',
        'output_dir': 'test3_output/',
        'plot': False,
        'name': 'test3',
        'text': True,
        'method': 'cluster'
    }
    test_dataset4 = {
        'data_file': 'test.txt',
        'output_dir': 'test4_output/',
        'plot': False,
        'name': 'test3',
        'text': True,
        'method': 'cluster'
    }
    test_dataset5 = {
        'data_file': 'test.txt',
        'output_dir': 'test5_output/',
        'plot': False,
        'name': 'test5',
        'text': True,
        'method': 'cluster'
    }
    test_dataset6 = {
        'data_file': 'test.txt',
        'output_dir': 'test6_output/',
        'plot': False,
        'name': 'test6',
        'text': True,
        'method': 'cluster'
    }
    test_dataset7 = {
        'data_file': 'test.txt',
        'output_dir': 'test7_output/',
        'plot': False,
        'name': 'test7',
        'text': True,
        'method': 'cluster'
    }
    test_dataset8 = {
        'data_file': 'test.txt',
        'output_dir': 'test8_output/',
        'plot': False,
        'name': 'test8',
        'text': True,
        'method': 'cluster'
    }
    test_dataset9 = {
        'data_file': 'test.txt',
        'output_dir': 'test9_output/',
        'plot': False,
        'name': 'test9',
        'text': True,
        'method': 'cluster'
    }
    test_dataset10 = {
        'data_file': 'test.txt',
        'output_dir': 'test10_output/',
        'plot': False,
        'name': 'test10',
        'text': True,
        'method': 'cluster'
    }
    test_dataset11 = {
        'data_file': 'test.txt',
        'output_dir': 'test11_output/',
        'plot': False,
        'name': 'test11',
        'text': True,
        'method': 'cluster'
    }
    test_dataset12 = {
        'data_file': 'test.txt',
        'output_dir': 'test12_output/',
        'plot': False,
        'name': 'test12',
        'text': True,
        'method': 'cluster'
    }


    r109_reduced_dataset = {
        'data_file': 'r109_reduced.txt',
        'output_dir': 'r109_reduced_output/',
        'plot': False,
        'name': 'r109_reduced',
        'text': False,
        'method': 'cluster'
    }
    # kernel.solve_and_plot([r109_reduced_dataset, ])


    # --- c201, r201, rc201 --- #
    c201_dataset = {
        'data_file': 'c201_mod.txt',
        'output_dir': 'c201_output/',
        'plot': False,
        'name': 'c201',
        'text': False,
        'method': 'cluster'
    }
    r201_dataset = {
        'data_file': 'r201_mod.txt',
        'output_dir': 'r201_output/',
        'plot': False,
        'name': 'r201',
        'text': False,
        'method': 'cluster'
    }
    rc201_dataset = {
        'data_file': 'rc201_mod.txt',
        'output_dir': 'rc201_output/',
        'plot': False,
        'name': 'rc201',
        'text': False,
        'method': 'cluster'
    }

    # kernel.solve_and_plot([c201_dataset])
    # kernel.solve_and_plot([r201_dataset])
    # kernel.solve_and_plot([rc201_dataset])


    # --- c104, r110, rc103 --- #
    c104_dataset = {
        'data_file': 'c104_mod.txt',
        'output_dir': 'c104_output/',
        'plot': False,
        'name': 'c104',
        'text': False,
        'method': 'cluster'
    }
    r110_dataset = {
        'data_file': 'r110_mod.txt',
        'output_dir': 'r110_output/',
        'plot': False,
        'name': 'r110',
        'text': False,
        'method': 'cluster'
    }
    rc103_dataset = {
        'data_file': 'rc103_mod.txt',
        'output_dir': 'rc103_output/',
        'plot': False,
        'name': 'rc103',
        'text': False,
        'method': 'cluster'
    }

    C1_2_4_dataset = {
        'data_file': 'C1_2_4_mod.txt',
        'output_dir': 'C1_2_4_output/',
        'plot': False,
        'name': 'C1_2_4',
        'text': False,
        'method': 'cluster'
    }
    R1_2_10_dataset = {
        'data_file': 'R1_2_10_mod.txt',
        'output_dir': 'R1_2_10_output/',
        'plot': False,
        'name': 'R1_2_10',
        'text': False,
        'method': 'cluster'
    }
    RC1_2_3_dataset = {
        'data_file': 'RC1_2_3_mod.txt',
        'output_dir': 'RC1_2_3_output/',
        'plot': False,
        'name': 'RC1_2_3',
        'text': False,
        'method': 'cluster'
    }

    C1_4_4_dataset = {
        'data_file': 'C1_4_4_mod.txt',
        'output_dir': 'C1_4_4_output/',
        'plot': False,
        'name': 'C1_4_4',
        'text': False,
        'method': 'cluster'
    }
    R1_4_10_dataset = {
        'data_file': 'R1_4_10_mod.txt',
        'output_dir': 'R1_4_10_output/',
        'plot': False,
        'name': 'R1_4_10',
        'text': False,
        'method': 'cluster'
    }
    RC1_4_3_dataset = {
        'data_file': 'RC1_4_3_mod.txt',
        'output_dir': 'RC1_4_3_output/',
        'plot': False,
        'name': 'RC1_4_3',
        'text': False,
        'method': 'cluster'
    }

    C1_6_4_dataset = {
        'data_file': 'C1_6_4_mod.txt',
        'output_dir': 'C1_6_4_output/',
        'plot': False,
        'name': 'C1_6_4',
        'text': False,
        'method': 'cluster'
    }
    R1_6_10_dataset = {
        'data_file': 'R1_6_10_mod.txt',
        'output_dir': 'R1_6_10_output/',
        'plot': False,
        'name': 'R1_6_10',
        'text': False,
        'method': 'cluster'
    }
    RC1_6_3_dataset = {
        'data_file': 'RC1_6_3_mod.txt',
        'output_dir': 'RC1_6_3_output/',
        'plot': False,
        'name': 'RC1_6_3',
        'text': False,
        'method': 'cluster'
    }

    dims_array = np.array([100, 200, 400, 600])
    k3_array = np.array([2.0, 10.0, 100.0, 1000.0])

    testing_datasets = [[c104_dataset, C1_2_4_dataset, C1_4_4_dataset, C1_6_4_dataset],
                        ]

    # testing_datasets = [[test_dataset1, test_dataset2, test_dataset3, test_dataset4],
    #                     [test_dataset5, test_dataset6, test_dataset7, test_dataset8],
    #                     [test_dataset9, test_dataset10, test_dataset11, test_dataset12]]

    for dataset_series in testing_datasets:
        for dataset in dataset_series:
            base_name = dataset['output_dir'][:len(dataset['output_dir']) - 1]
            for k3 in k3_array:
                kernel = Kernel(k3)
                dataset['output_dir'] = base_name + '_' + str(int(k3)) + '/'
                kernel.solve_and_plot([dataset, ])

    # statistics = Statistics(testing_datasets, dims_array, k3_array)
    # different_k3_arr = statistics.collect_time_data()
    # plotter.plot_data(testing_datasets, dims_array, k3_array, different_k3_arr, xlabel='Customers number', ylabel='Time execution')
    #
    # different_k3_arr_dist, different_k3_arr_wait_time, different_k3_arr_late_time, different_k3_arr_eval = statistics.collect_evaluation()
    # plotter.plot_data(testing_datasets, dims_array, k3_array, different_k3_arr_dist, xlabel='Customers number', ylabel='Total distance')
    # plotter.plot_data(testing_datasets, dims_array, k3_array, different_k3_arr_wait_time, xlabel='Customers number', ylabel='Wait time')
    # plotter.plot_data(testing_datasets, dims_array, k3_array, different_k3_arr_late_time, xlabel='Customers number', ylabel='Late time')
    # plotter.plot_data(testing_datasets, dims_array, k3_array, different_k3_arr_eval, xlabel='Customers number', ylabel='Total evaluation')
