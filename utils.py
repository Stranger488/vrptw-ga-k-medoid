import numpy as np

from functools import wraps
from time import time


def timing(f):

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result

    return wrap


class Utils:
    def __init__(self):
        # Cost for distance per unit
        self.c_D = 1.0

        # Cost for wait time_cluster per unit
        self.c_T = 1.0

        # Cost for late time_cluster per unit
        self.c_L = 1.5

    def read_standard_dataset(self, dataset, points_dataset, tws_all, service_time_all):
        for i in range(dataset.shape[0]):
            tws_all = np.concatenate((tws_all, [[dataset['READY_TIME'][i],
                                                 dataset['DUE_DATE'][i]]]), axis=0)

            service_time_all = np.concatenate((service_time_all, [[dataset['SERVICE_TIME'][i]]]), axis=0)

            points_dataset = np.concatenate((points_dataset, [[dataset['XCOORD'][i],
                                                               dataset['YCOORD'][i]]]), axis=0)

        return points_dataset, tws_all, service_time_all

    def evaluate_solution(self, tsptw_results, eval_method='default'):
        total_dist = 0.0
        wait_time = 0.0
        late_time = 0.0

        for result in tsptw_results:
            total_dist += result['Distance'][len(result) - 1]
            wait_time += result['Wait_Time'][len(result) - 1]
            late_time += result['Late_Time'][len(result) - 1]

        if eval_method == 'by_distance':
            f = open("file.txt", "a")
            f.write("total late time_cluster: {}\n".format(late_time))
            f.write("total wait time_cluster: {}\n\n".format(wait_time))
            f.close()
            # print("total late time_cluster: {}".format(late_time))
            return total_dist

        return self.c_D * total_dist + self.c_T * wait_time + self.c_L * late_time
