import datetime as dt
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import huey, get_task_logging_adapter
from core.db.data_access_facade_basic import DataAccessFacadeBasic
from core.distances_and_times.distance_time_modification import (
    calculate_time_traffic_factor,
    fit_function,
    get_route_function_approximation,
)
from core.utils.id_to_index_mapper import IdToIndexMapper

base_logger = logging.getLogger("fleet_manager")


@huey.task(context=True)
def run_distance_modification_approximation(
    zone_id: str,
    cell_ids: list[str],
    today: dt.date,
    approx_func: callable = fit_function,
    start_date: dt.date | None = None,
    output_zone_id: str | None = None,
    task=None,
    finished: bool = False,
):
    """
    Create and save route distance and time arrays.
    approx_func: distance modifying function F such that F(road distance matrix) -> rent distance matrix.
    """
    log = get_task_logging_adapter(base_logger, task)
    log.info(f"Begin distance approximation for zone {zone_id} and date {today}")
    min_distance_meters = 1000  # shorter rents will be filtered out
    max_hours_for_rent = 2  # longer rents will be filtered out
    n_weeks_back = 25  # set period of time for data aggregation
    bin_size = 500  # bins for real distances, ex. 0-500, 500-1000

    daf: DataAccessFacadeBasic = DataAccessFacadeBasic()
    id_mapper = IdToIndexMapper(original_ids=cell_ids)

    lower_date = start_date or dt.date(
        today.year, today.month, today.day
    ) - dt.timedelta(weeks=n_weeks_back)

    log.info(f"loading rents for date range {lower_date} - {today}")
    rents: pd.DataFrame = daf.find_rents_frame(zone_id, lower_date, today)
    rents = (
        rents.loc[
            (rents["distance"] > min_distance_meters)
            & (rents["user_time_minutes"] < max_hours_for_rent * 60)
        ]
        .sort_values(by="start_date")
        .reset_index(drop=True)
    )

    rents.start_date = pd.to_datetime(rents.start_date)

    log.info("mapping rents cell ids")
    rents.start_cell_id = id_mapper.map_ids_to_indices(rents.start_cell_id)
    rents.end_cell_id = id_mapper.map_ids_to_indices(rents.end_cell_id)
    log.info("loading distance and time matrices")
    distance_matrix: np.array = daf.find_distance_cell_cell_array(zone_id)
    time_matrix: np.array = daf.find_time_cell_cell_array(zone_id)

    log.info("running distance approximation")
    popt: np.array = get_route_function_approximation(
        rents=rents,
        distance_matrix=distance_matrix,
        bin_size=bin_size,
        approx_func=approx_func,
    )
    log.info(f"calculated coefficients: {popt}")
    log.info("applying function to distances and times")
    route_distances = approx_func(distance_matrix, *popt)
    route_times = approx_func(time_matrix, *popt)

    log.info("saving route distances and times")
    output_zone_id = output_zone_id or zone_id
    daf.save_route_distance_cell_cell_array(
        zone_id=output_zone_id, route_distance_cell_cell_array=route_distances
    )
    daf.save_route_time_cell_cell_array(
        zone_id=output_zone_id, route_time_cell_cell_array=route_times
    )
    log.info(f"End distance approximation")
    return {"finished": True}


@huey.task(context=True)
def run_traffic_factors_task(
    zone_id: str,
    today: dt.datetime,
    start_date: dt.date | None = None,
    output_zone_id: str | None = None,
    task=None,
    finished: bool = False,
):
    """
    Create rents list and get time traffic factors.
    """
    log = get_task_logging_adapter(base_logger, task)
    log.info(f"Begin calculating time traffic factor")
    n_weeks_back = 1
    max_hours_for_rent = 2
    min_distance_meters = 1000
    start_night_hour = 21
    end_night_hour = 4

    daf = DataAccessFacadeBasic()

    lower_date = start_date or dt.date(
        today.year, today.month, today.day
    ) - dt.timedelta(weeks=n_weeks_back)

    log.info(f"loading rents for date range {lower_date} - {today}")
    rents: pd.DataFrame = daf.find_rents_frame(zone_id, lower_date, today)
    rents = (
        rents.loc[
            (rents["distance"] > min_distance_meters)
            & (rents["user_time_minutes"] < max_hours_for_rent * 60)
        ]
        .sort_values(by="start_date")
        .reset_index(drop=True)
    )

    rents.start_date = pd.to_datetime(rents.start_date)

    rents["weekday"] = rents.start_date.apply(lambda d: d.weekday())
    rents["start_hour"] = rents.start_date.apply(lambda d: d.hour)

    log.info(
        f"calculating time traffic factor for start night hour {start_night_hour} and end night hour {end_night_hour}"
    )
    ttf: np.array = calculate_time_traffic_factor(
        rents=rents, start_night_hour=start_night_hour, end_night_hour=end_night_hour
    )
    output_zone_id = output_zone_id or zone_id
    log.info("saving time traffic factor array")
    daf.save_time_traffic_factor_array(
        zone_id=output_zone_id, time_traffic_factor_array=ttf
    )
    log.info(f"End calculating time traffic factor")
    return {"finished": True}