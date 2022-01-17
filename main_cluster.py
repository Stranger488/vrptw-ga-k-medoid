import argparse

from mpl_toolkits.mplot3d.axis3d import Axis

# fix wrong z-offsets in 3d plot
from launcher_cluster import LauncherCluster


def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0, 0, 1.0 / 4]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs


def set_options():
    # fix wrong z-offsets in 3d plot
    if not hasattr(Axis, "_get_coord_info_old"):
        Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--in_dataset_series', action='store', type=str,
                            help='Input file with dataset series to solve')
    arg_parser.add_argument('--mode', action='store_true',
                            help='Solving with procs 1 to os.cpu_count() or os.cpu_count()')

    arguments = arg_parser.parse_args()

    return arguments


def main(arguments):
    launcher = LauncherCluster(in_dataset_series=arguments.in_dataset_series, mode=arguments.mode)
    launcher.launch()


if __name__ == '__main__':
    args = set_options()

    main(args)
