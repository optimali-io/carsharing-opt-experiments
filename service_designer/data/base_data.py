"""
Module contains data model for Simulator DayByDay.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from core.utils.nptools import assert_ndarray

log = logging.getLogger("service_designer")


class BaseData:
    """
    Data model for Simulator DayByDay - collect data matrix's of base input data of Simulator.

    :param data_config: data configuration parameters collected in dictionary
    :param days: number of days in week
    :param hours: number of hours in day
    """

    def __init__(self, data_config: Dict, days: int = 7, hours: int = 24):

        self.data_config: Dict = data_config
        self.days: int = days
        self.hours: int = hours
        self.cell_ids: List[str] = data_config["base_zone"]["cell_ids"]
        self.cells: int = len(self.cell_ids)
        self.here_distance: np.array = None
        self.distance_to_station: np.array = None
        self.stations: pd.DataFrame = None

        self.rent_demand: np.array = None
        self.rent_direction: np.array = None

        self.route_distance: np.array = None  # Here distance array after distance_modification_approximation
        self.route_time: np.array = None  # Here time array after distance_modification_approximation
        self.time_traffic_factor: np.array = None

    def assert_arrays(self) -> None:
        """
        Assert arrays dimensions and types.
        """

        h = self.days * self.hours
        c = self.cells
        assert_ndarray(self.here_distance, (c, c), np.uint32, "here_distance")
        assert_ndarray(self.distance_to_station, (c,), np.float64, "distance_to_station")

        assert_ndarray(self.rent_demand, (h, c), np.float32, "rent_demand")
        assert_ndarray(self.rent_direction, (h, c, c), np.float32, "rent_direction")

        assert_ndarray(self.route_distance, (c, c), np.float64, "route_distance")
        assert_ndarray(self.route_time, (c, c), np.float64, "route_time")
        assert_ndarray(self.time_traffic_factor, (self.days, self.hours), np.float64, "time_traffic_factor")

    def assert_rent_direction(self) -> None:
        """
        Assert values of the rent direction array.
        """

        d = self.rent_direction
        log.info("Evaluate directions")
        log.info(f"Directions min avg max {d.min()} {d.mean()} {d.max()}")
        for c in range(self.cells):
            s = d[:, c, :].sum()
            if s == 0:
                tcs = d[:, :, c].sum()
                log.info(f"No directions from zone cell {c}, to cell sum is {tcs}")
            else:
                pass

    def memory_size_gb(self) -> float:
        """
        Calculate memory usage of this `BaseData` instance in GB.

        return: memory usage in GB
        """
        bn = 0
        bn += self.here_distance.nbytes
        bn += self.distance_to_station.nbytes
        bn += self.rent_demand.nbytes
        bn += self.rent_direction.nbytes
        bn += self.route_distance.nbytes
        bn += self.route_time.nbytes
        bn += self.time_traffic_factor.nbytes
        return bn / 1024 / 1024 / 1024
