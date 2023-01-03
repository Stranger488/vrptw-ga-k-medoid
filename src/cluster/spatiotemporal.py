import math

import numpy as np
from scipy.integrate import quad

from src.common.utils import calc_euclidian_dist_all


class Spatiotemporal:
    def __init__(self, dataset, tws, service_time,
                 k1=1.0, k2=1.5, k3=2.0, alpha1=0.5, alpha2=0.5):
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3

        self._alpha1 = alpha1
        self._alpha2 = alpha2

        self._tws_all = tws
        self.MAX_TW = self._get_max_tw()

        self._service_time_all = service_time

        self._points_all = dataset

        length = self._points_all.shape[0]
        self.euclidian_dist_all = np.zeros((length, length))
        self.temporal_all = np.zeros((length, length))
        self.temporal_dist_all = np.zeros((length, length))

        self._min_s_m_neq_n = math.inf
        self._min_t_m_neq_n = math.inf
        self._max_s_m_neq_n = -math.inf
        self._max_t_m_neq_n = -math.inf

        self.spatiotemporal_dist_all = np.zeros((length, length))

    def fill_temporal_dist_all(self):
        length = self.temporal_dist_all[0].size
        self.temporal_dist_all = np.array(
            [[self._D_temporal_integr(i, j) for j in range(length)] for i in range(length)])

    def fill_spatiotemporal_dist_all(self):
        self.spatiotemporal_dist_all = self._D_spatiotemporal_vect(self.euclidian_dist_all, self.temporal_dist_all)

    def calculate_all_distances(self):
        # fill all euclidian distances between points
        self.euclidian_dist_all = calc_euclidian_dist_all(self._points_all)

        # init numerically the same temporal distance as spatial
        self.temporal_all = np.copy(self.euclidian_dist_all)

        # calculate all temporal distances and normalize it
        self.fill_temporal_dist_all()
        self.temporal_dist_all = self._norm_temporal_dist_all()

        # find necessary values for calculating spatiotemporal distances
        self._min_s_m_neq_n = self._get_min(self.euclidian_dist_all)
        self._min_t_m_neq_n = self._get_min(self.temporal_dist_all)
        self._max_s_m_neq_n = self._get_max(self.euclidian_dist_all)
        self._max_t_m_neq_n = self._get_max(self.temporal_dist_all)

        # calculate spatiotemporal distances between all points
        self.fill_spatiotemporal_dist_all()

        return self.spatiotemporal_dist_all

    def _get_max_tw(self):
        return np.subtract(self._tws_all[:, 1], self._tws_all[:, 0]).max()

    def _get_min(self, data):
        return data[1:, 1:][data[1:, 1:] != 0].min()

    def _get_max(self, data):
        return data[1:, 1:][data[1:, 1:] != 0].max()

    def _Sav1(self, t_cur, i, j):
        return self._k2 * t_cur + self._k1 * self._tws_all[j][1] - (self._k1 + self._k2) * self._tws_all[j][0]

    def _Sav2(self, t_cur, i, j):
        return -self._k1 * t_cur + self._k1 * self._tws_all[j][1]

    def _Sav3(self, t_cur, i, j):
        return -self._k3 * t_cur + self._k3 * self._tws_all[j][1]

    def _D_temporal_integr(self, i, j):
        if i != j:
            customer_i_a_s = self._tws_all[i][0] + self._service_time_all[i][0] + self.temporal_all[i][j]
            customer_i_b_s = self._tws_all[i][1] + self._service_time_all[i][0] + self.temporal_all[i][j]

            min_1 = min(customer_i_a_s, self._tws_all[j][0])
            max_1 = min(customer_i_b_s, self._tws_all[j][0])
            integr_1 = quad(self._Sav1, min_1, max_1, args=(i, j))[0]

            min_2 = min(
                max(customer_i_a_s, self._tws_all[j][0]),
                self._tws_all[j][1]
            )
            max_2 = max(
                min(customer_i_b_s, self._tws_all[j][1]),
                self._tws_all[j][0]
            )
            integr_2 = quad(self._Sav2, min_2, max_2, args=(i, j))[0]

            min_3 = max(customer_i_a_s, self._tws_all[j][1])
            max_3 = max(customer_i_b_s, self._tws_all[j][1])
            integr_3 = quad(self._Sav3, min_3, max_3, args=(i, j))[0]

            return self._k1 * self.MAX_TW - (integr_1 + integr_2 + integr_3) / (customer_i_b_s - customer_i_a_s)

        return 0.0

    def _norm_temporal_dist_all(self):
        # element-wise minimum of initial and transposed matrix to compare [i, j] and [j, i]
        return np.fmin(self.temporal_dist_all, self.temporal_dist_all.transpose())

    def _D_spatiotemporal_vect(self, euclidian, temporal):
        # TODO: is there a need to check i != j?
        null_indices = np.where(euclidian == 0.0)
        result = (self._alpha1 * (euclidian - self._min_s_m_neq_n) / (
                self._max_s_m_neq_n - self._min_s_m_neq_n) + self._alpha2 * (temporal - self._min_t_m_neq_n) / (
                          self._max_t_m_neq_n - self._min_t_m_neq_n))
        result[null_indices] = 0.0
        return result
