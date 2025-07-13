import logging

import numpy as np

from core.utils.nptools import convert_trip_density_to_trip_distribution

log = logging.getLogger(__name__)


def generate_direction_realizations(direction_hcc: np.ndarray) -> np.ndarray:
    """
    Generate rent direction realization of rent direction density.

    :param direction_hcc: rent direction density
    :type direction_hcc: numpy.ndarray, shape=(H, C, C)

    :return: samples of realization of rent direction density
    :rtype: numpy.ndarray, shape=(H, C, S)
    """

    probability_values = np.copy(direction_hcc)
    _fix_trip_destination_density(probability_values)
    trip_distribution = _prepare_trip_distribution(probability_values)
    trip_destination_hour_cell_sample = _prepare_trip_destination(trip_distribution)
    return trip_destination_hour_cell_sample


def _fix_trip_destination_density(trip_destination_density_hour_cell_cell: np.ndarray) -> None:
    """
    In case of missing probability density increase chance of rent with end cell equal to start cell.

    :param trip_destination_density_hour_cell_cell: rent direction density
    :type trip_destination_density_hour_cell_cell: numpy.ndarray, shape=(H, C, C)

    :return: nothing
    :rtype: None
    """

    hours_nbr, cells_nbr, _ = trip_destination_density_hour_cell_cell.shape
    for hour in range(hours_nbr):
        for src_cell in range(cells_nbr):
            missing_probability = 1.0 - trip_destination_density_hour_cell_cell[hour, src_cell, :].sum()
            if missing_probability > 0:
                trip_destination_density_hour_cell_cell[hour, src_cell, src_cell] += missing_probability


def _prepare_trip_distribution(probability_values: np.ndarray) -> np.ndarray:
    """
    Generate rent direction distribution from rent direction density.

    :param probability_values: rent direction density
    :type probability_values: numpy.ndarray, shape=(H, C, C)

    :return: rent direction distribution
    :rtype: numpy.ndarray, shape=(H, C, C)
    """

    log.info("_prepare_trip_distribution")
    trip_distribution = np.zeros(probability_values.shape, dtype=np.float32)
    hours = probability_values.shape[0]
    for h in range(hours):
        trip_distribution[h, :, :] = convert_trip_density_to_trip_distribution(probability_values[h, :, :])
    return trip_distribution


def _prepare_trip_destination(trip_distribution: np.ndarray) -> np.ndarray:
    """
    Generate rent direction realization samples of rent direction distribution.

    :param trip_distribution: rent direction density
    :type trip_distribution: numpy.ndarray, shape=(H, C, C)

    :return: samples of realization of rent direction density
    :rtype: numpy.ndarray, shape=(H, C, S)
    """

    log.info("_prepare_trip_destination")
    hours, cells, _ = trip_distribution.shape
    destinations = 10_000
    max_cell_idx = cells - 1
    trip_destination = np.zeros((hours, cells, destinations), dtype=np.uint16)
    step = 1.0 / destinations
    x = np.arange(0.0, 1.0, step)
    for c in range(cells):
        for h in range(hours):
            dest_cells = np.searchsorted(trip_distribution[h, c, :], x)
            dest_cells = np.minimum(dest_cells, max_cell_idx)
            np.random.shuffle(dest_cells)
            trip_destination[h, c] = dest_cells
    return trip_destination
