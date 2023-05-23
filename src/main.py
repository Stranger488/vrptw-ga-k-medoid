from mpl_toolkits.mplot3d.axis3d import Axis

from src.common.utils import set_options
from src.vrptw.launcher import Launcher


# TODO: должно быть в main'e?
def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0, 0, 1.0 / 4]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs


def main(arguments):
    launcher = Launcher(launch_entries=arguments.launch_entries,
                        plot_stats=arguments.plot_stats,
                        plot_solutions=arguments.plot_solutions,
                        solve_cluster=arguments.solve_cluster,
                        solve_tsptw=arguments.solve_tsptw)
    launcher.launch()


if __name__ == '__main__':
    if not hasattr(Axis, "_get_coord_info_old"):
        Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new

    args = set_options()
    main(args)
