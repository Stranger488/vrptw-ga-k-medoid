import os


class VRPTWPathHolder:
    def __init__(self, vrptw_entry_id, solve_cluster, solve_tsptw):
        self.BASE_DIR = os.path.abspath(os.curdir)

        combine_modes = solve_cluster + '_' + solve_tsptw
        self.BASE_CUR_SOLUTION_DIR = self.BASE_DIR + '/vrptw_result/' + vrptw_entry_id + '/' + combine_modes
        self.CLUSTER_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/cluster_result/'
        self.TSPTW_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/tsptw_result/'
        self.EVALUATION_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/evaluation/'
        self.PLOT_STATS_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/plot_stats/'
        self.PLOT_SOLUTIONS_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/plot_solutions/'
