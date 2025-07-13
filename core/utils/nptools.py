from typing import Tuple

import numpy as np


def assert_ndarray(a: np.ndarray, shape: Tuple[int, ...], dtype: np.dtype, name: str) -> None:
    """
    Assert that ndarray has expected shape and type.

    :param a: array to check
    :type a: numpy.ndarray

    :param shape: expected shape
    :type shape: Tuple[int, ...]

    :param dtype: expected dtype
    :type dtype: numpy.dtype

    :param name: array name used in error message
    :type name: str

    :return: nothing
    :rtype: None
    """

    assert a.shape == shape, f"Expected {name} ndarray shape is {shape} while actual is {a.shape}"
    assert a.dtype == dtype, f"Expected {name} ndarray data type is {dtype} while actual is {a.dtype}"


def convert_trip_density_to_trip_distribution(density_cc: np.ndarray) -> np.ndarray:
    """
    Build direction distribution from direction density.

    :param density_cc: direction density
    :type density_cc: numpy.ndarray, shape=(C, C)

    :return: direction distribution
    :rtype: numpy.ndarray, shape=(C, C)
    """

    distribution = np.cumsum(density_cc, axis=1)
    return distribution


def rearrange_from_day_hour_cell_to_hour_cell(arr: np.ndarray, days: int, hours: int, cells: int) -> np.ndarray:
    """
    Create 2d array from 3d array by flattening "days" (first) axis.
    :param arr: np.ndarray, must be 3d
    :param days: number of days
    :param hours: number of hours
    :param cells: number of cells
    """
    arr2 = np.reshape(arr, (days * hours, cells))
    return arr2


def rearrange_from_day_hour_cell_cell_to_hour_cell_cell(
    arr: np.ndarray, days: int, hours: int, cells: int
) -> np.ndarray:
    """
    Create 3d array from 4d array by flattening "days" (first) axis.
    :param arr: np.ndarray, must be 4d
    :param days: number of days
    :param hours: number of hours
    :param cells: number of cells
    """
    arr2 = np.reshape(arr, (days * hours, cells, cells))
    return arr2
