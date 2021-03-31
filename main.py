from kernel import Kernel

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
    kernel = Kernel()

    test_dataset = {
        'data_file': 'test.txt',
        'output_dir': 'test_output/',
        'plot': True,
        'name': 'test',
        'text': True,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    # kernel.solve_and_plot([test_dataset, ])


    r109_reduced_dataset = {
        'data_file': 'r109_reduced.txt',
        'output_dir': 'r109_reduced_output/',
        'plot': True,
        'name': 'r109_reduced',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    # solve_and_plot([r109_reduced_dataset, ])


    # --- c104, r110, rc103 --- #
    # --- c201, r201, rc201 --- #

    c104_dataset = {
        'data_file': 'c104_mod.txt',
        'output_dir': 'c104_output/',
        'plot': False,
        'name': 'c104',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    r110_dataset = {
        'data_file': 'r110_mod.txt',
        'output_dir': 'r110_output/',
        'plot': False,
        'name': 'r110',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    rc103_dataset = {
        'data_file': 'rc103_mod.txt',
        'output_dir': 'rc103_output/',
        'plot': False,
        'name': 'rc103',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }

    kernel.solve_and_plot([c104_dataset])
    # solve_and_plot([r110_dataset])
    # solve_and_plot([rc103_dataset])


    c201_dataset = {
        'data_file': 'c201_mod.txt',
        'output_dir': 'c201_output/',
        'plot': False,
        'name': 'c201',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    r201_dataset = {
        'data_file': 'r201_mod.txt',
        'output_dir': 'r201_output/',
        'plot': False,
        'name': 'r201',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    rc201_dataset = {
        'data_file': 'rc201_mod.txt',
        'output_dir': 'rc201_output/',
        'plot': False,
        'name': 'rc201',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }

    # solve_and_plot([c201_dataset])
    # solve_and_plot([r201_dataset])
    # solve_and_plot([rc201_dataset])


    C1_10_2_dataset = {
        'data_file': 'C1_10_2_mod.txt',
        'output_dir': 'C1_10_2_output/',
        'plot': False,
        'name': 'C1_10_2',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    R1_10_1_dataset = {
        'data_file': 'R1_10_1_mod.txt',
        'output_dir': 'R1_10_1_output/',
        'plot': False,
        'name': 'R1_10_1',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }

    # solve_and_plot([C1_10_2_dataset])
    # solve_and_plot([R1_10_1_dataset])
