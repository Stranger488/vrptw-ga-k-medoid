from numpy.core.records import ndarray

from src.cluster.spatiotemporal import Spatiotemporal


class ClusterResultEntry:
    def __init__(self, result: ndarray, res_dataset: ndarray,
                 res_tws: ndarray, spatiotemporal: Spatiotemporal,
                 dataset_reduced: ndarray):
        self.result = result
        self.res_dataset = res_dataset
        self.res_tws = res_tws
        self.spatiotemporal = spatiotemporal
        self.dataset_reduced = dataset_reduced
