import os


class VRPTWPathHolder:
    def __init__(self, vrptw_entry_id, mode_path):
        self.BASE_DIR = os.path.abspath(os.curdir)

        self.BASE_CUR_SOLUTION_DIR = self.BASE_DIR + '/vrptw_result/' + vrptw_entry_id + '/' + mode_path
        self.CLUSTER_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/cluster_result/'
        self.TSPTW_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/tsptw_result/'
        self.EVALUATION_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/evaluation/'
        self.PLOT_STATS_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/plot_stats/'
        self.PLOT_SOLUTIONS_OUTPUT = self.BASE_CUR_SOLUTION_DIR + '/plot_solutions/'
