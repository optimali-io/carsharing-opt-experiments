import logging

import numpy as np


log = logging.getLogger(__name__)


def distribute_demand_decimal_part(demand_hc: np.ndarray) -> np.ndarray:
    """
    Generate rent demand realization of rent demand expected value. This is not mathematically correct implementation.
    Integral part is the same as integral part of expected value. Decimal part is chosen with probability proportional
    to decimal part value.

    :param demand_hc: rent demand expected value
    :type demand_hc: numpy.ndarray, shape=(H, C)

    :return: samples of realization of rent demand expected value
    :rtype: numpy.ndarray, shape=(S, H, C)
    """

    cells_nbr = demand_hc.shape[1]
    default_demand_hc = demand_hc
    demand_values_int16 = default_demand_hc.astype(dtype=np.uint16)
    demand_values_rest = default_demand_hc - demand_values_int16
    all_sum = np.sum(default_demand_hc)
    small_sum = np.sum(demand_values_rest)
    log.info("SMALL DEMAND PERCENT " + str(int(100 * small_sum / all_sum)) + " percent")
    log.info("AVERAGE DEMAND PER CELL " + str(all_sum / cells_nbr))
    demand_values_rest_occurrence = _calculate_weighted_rest_occurrence(demand_values_rest, 1000)
    demand_sample_hour_cell = np.add(demand_values_int16, demand_values_rest_occurrence)
    return demand_sample_hour_cell


def _calculate_weighted_rest_occurrence(demand_rest: np.ndarray, number: int) -> np.ndarray:
    """
    Generate rent demand decimal part realization of rent demand expected value decimal part. This is not mathematically
    correct implementation.

    :param demand_rest: rent demand expected value decimal part
    :type demand_rest: numpy.ndarray, shape=(H, C)

    :return: samples of realization of rent demand expected value decimal part
    :rtype: numpy.ndarray, shape=(S, H, C)
    """

    log.info("_calculate_weighted_rest_occurrence")
    hours, cells = demand_rest.shape
    result_shc = np.zeros((number, hours, cells), dtype=np.uint16)

    rest_sum = np.sum(demand_rest)
    if rest_sum > 0:
        rest_prob_hc = demand_rest / rest_sum
        rest_prob_d1 = np.reshape(rest_prob_hc, -1)
        log.info("rest car sum " + str(rest_sum))

        for n in range(number):
            if n % 100 == 0:
                log.info("generate rest demand weighted random placement progress " + str(n) + " of " + str(number))
            sample_indexes = np.random.choice(len(rest_prob_d1), int(rest_sum), p=rest_prob_d1)
            sample_d1 = np.bincount(sample_indexes, minlength=(cells * hours))
            sample_d1 = sample_d1.astype(np.uint16)
            sample_hc = np.reshape(sample_d1, (hours, cells))
            result_shc[n, :, :] = sample_hc

    return result_shc
