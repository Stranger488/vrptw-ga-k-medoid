import argparse

from launcher import Launcher

from mpl_toolkits.mplot3d.axis3d import Axis


# fix wrong z-offsets in 3d plot
def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0, 0, 1.0 / 4]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs


def main(arguments):
    launcher = Launcher(in_dataset_series=arguments.in_dataset_series, is_solve=arguments.solve,
                        plot_stats=arguments.plot_stats, mode=arguments.mode)
    launcher.launch()


def set_options():
    # fix wrong z-offsets in 3d plot
    if not hasattr(Axis, "_get_coord_info_old"):
        Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--in_dataset_series', action='store', type=str,
                            help='Input file with dataset series to solve')
    arg_parser.add_argument('--solve', action='store_true', help='Solving all datasets or not')
    arg_parser.add_argument('--plot_stats', action='store', type=str,
                            help='Plot statistics, value is stats that will be displayed')
    arg_parser.add_argument('--mode', action='store', type=str,
                            help='Launch only cluster, tsptw, or both steps')

    arguments = arg_parser.parse_args()

    return arguments


if __name__ == '__main__':
    args = set_options()

    main(args)
