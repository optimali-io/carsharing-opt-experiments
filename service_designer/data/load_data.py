"""
Module contains data model loading functions.
"""

import logging

import numpy as np
import pandas as pd

from core.db.data_access_facade_basic import DataAccessFacadeBasic
from core.utils.nptools import (
    rearrange_from_day_hour_cell_cell_to_hour_cell_cell,
    rearrange_from_day_hour_cell_to_hour_cell,
)
from service_designer.data.base_data import BaseData
from service_designer.data.data_config import ExperimentConfig

log = logging.getLogger("service_designer")


def load_base_data(experiment_config: ExperimentConfig) -> BaseData:
    """Loading arrays to BaseData model."""
    bd = BaseData(experiment_config.data_config.model_dump())
    log.info(f"Loading BaseData of {bd.data_config['name']}")

    base_zone_id = experiment_config.data_config.base_zone.name
    data_config_zone_id = experiment_config.get_experiment_zone_id()

    daf = DataAccessFacadeBasic()

    bd.stations: pd.DataFrame = daf.find_gas_stations_frame_sd(zone_id=data_config_zone_id)

    bd.here_distance = daf.find_distance_cell_cell_array(base_zone_id)
    bd.distance_to_station = daf.find_distance_cell_station_array(data_config_zone_id)

    bd.rent_demand = daf.find_demand_prediction_array(data_config_zone_id, from_date=experiment_config.data_config.end_date)
    directions_week = []
    for weekday in range(7):
        directions_week.append(
            daf.find_directions_array(data_config_zone_id, weekday=weekday)
        )
    bd.rent_direction = np.array(directions_week)

    bd.route_distance = daf.find_route_distance_cell_cell_array(zone_id=data_config_zone_id)
    bd.route_time = daf.find_route_time_cell_cell_array(zone_id=data_config_zone_id)
    bd.time_traffic_factor = daf.find_time_traffic_factor_array(zone_id=data_config_zone_id)

    log.info(f"max distance: here {bd.here_distance.max()}\t route {bd.route_distance.max()}")

    bd.rent_demand = rearrange_from_day_hour_cell_to_hour_cell(bd.rent_demand, bd.days, bd.hours, bd.cells)
    bd.rent_demand = bd.rent_demand.astype(np.float32)

    bd.rent_direction = rearrange_from_day_hour_cell_cell_to_hour_cell_cell(
        bd.rent_direction, bd.days, bd.hours, bd.cells
    )
    _fix_rent_direction_density(bd.rent_direction)

    log.info(f"Loaded BaseData of {bd.data_config['name']}, size is {bd.memory_size_gb():0.2f} GB")
    return bd


def _fix_rent_direction_density(direction_hcc):
    hours_nbr = direction_hcc.shape[0]
    cells_nbr = direction_hcc.shape[1]
    for hour in range(hours_nbr):
        for src_cell in range(cells_nbr):
            missing_probability = 1.0 - direction_hcc[hour, src_cell, :].sum()
            if missing_probability > 0:
                direction_hcc[hour, src_cell, src_cell] += missing_probability
