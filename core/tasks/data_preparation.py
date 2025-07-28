import datetime as dt
import logging
from copy import copy
from typing import Dict, List

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from config import huey, get_task_logging_adapter
from core.db.data_access_facade_basic import (
    DataAccessFacadeBasic,
    create_rents_and_appstarts_frames,
)
from core.demand.blur_demand import blur_demand_history
from core.demand.demand_prediction import predict_7_days_of_demand
from core.demand.historical_demand import GetHistoricalDemand
from core.directions.historical_directions import GetHistoricalDirections
from core.distances_and_times.get_dist_time_here_matrices import (
    DistanceTimeMatrixCreator,
)
from fleet_manager.fastsimulator.fast_simulator import FastSimulator
from fleet_manager.revenueestimation.revenue_estimation import (
    RevenueEstimation,
    generate_revenue_estimation,
)

base_logger = logging.getLogger(__name__)


@huey.task(context=True)
def calculate_historical_demand(
    today: dt.date,
    zone_id: str,
    cell_ids: list[str],
    output_zone_id: str | None = None,
    task=None,
):
    """
    Calculate historical demand matrix for given Zone and date and store it.
    """
    log = get_task_logging_adapter(base_logger, task)
    print(f"Begin calculate historical demand for zone {zone_id} and date {today}")
    upper_date = dt.datetime(today.year, today.month, today.day, 0)

    lower_date = upper_date - dt.timedelta(
        days=2
    )  # we need two days to catch unfinished rents
    date_day_before = pd.Timestamp(upper_date - dt.timedelta(days=1)).tz_localize(
        "utc"
    )  # needed to filter demand points after calculation

    print(f"loading rents and appstarts")
    rents, appstarts = create_rents_and_appstarts_frames(
        lower_date.date(), upper_date.date(), zone_id, cell_ids
    )

    gd = GetHistoricalDemand(rents_df=rents, appstarts_df=appstarts)
    print(f"get demand")
    df = gd.get_demand()
    df = df.loc[df["time"] > date_day_before]

    n_cells = len(cell_ids)
    print(f"aggregate demand")
    demand_history_hc = GetHistoricalDemand.aggregate_demand(df, n_cells)

    for i in range(24):
        print(f"demand sum for hour {i}: {np.sum(demand_history_hc[i])}")

    print("save demand history array")
    daf: DataAccessFacadeBasic = DataAccessFacadeBasic()
    output_zone_id = output_zone_id or zone_id
    daf.save_demand_history_array(output_zone_id, today, demand_history_hc)
    print(f"End calculate historical demand")

    return True


@huey.task(context=True)
def calculate_demand_prediction(
    today: dt.date,
    zone_id: str,
    blur_factor: float,
    blur_distance: float,
    output_zone_id: str | None = None,
    from_date: dt.date | None = None,
    task=None,
):
    """
    Calculate demand prediction matrix for given Zone and date.
    """

    log = get_task_logging_adapter(base_logger, task)
    print(f"Begin calculate demand prediction for zone {zone_id} and date {today}")
    daf: DataAccessFacadeBasic = DataAccessFacadeBasic()

    from_date = from_date or today - dt.timedelta(days=49)
    to_date = today - dt.timedelta(days=1)
    print(f"find demand history array for date range {from_date} - {to_date}")
    demand_history_dhc = daf.find_demand_history_array(
        output_zone_id or zone_id, from_date, to_date
    )

    print("load distance cell cell array")
    distance_cc = daf.find_distance_cell_cell_array(zone_id)

    print("blur demand")
    demand_history_blurred_dhc = blur_demand_history(
        demand_history_dhc, distance_cc, blur_distance, blur_factor
    )

    print("predicting 7 days of demand")
    demand_prediction_dhc = predict_7_days_of_demand(demand_history_blurred_dhc)
    for day in range(7):
        for hour in range(24):
            print(
                f"demand prediction sum {np.sum(demand_prediction_dhc[day, hour])} for weekday {day} and hour {hour}"
            )
    print("saving demand prediction array")
    output_zone_id = output_zone_id or zone_id
    daf.save_demand_prediction_array(output_zone_id, today, demand_prediction_dhc)
    print(f"End calculate demand prediction")


@huey.task(context=True)
def calculate_demand_by_weekday(
    today: dt.date,
    zone_id: str,
    cell_ids: list[str],
    blur_factor: float,
    blur_distance: float,
    output_zone_id: str | None = None,
    from_date: dt.date | None = None,
    task=None,
):
    """
    Calculate demand by weekday matrix for given Zone and date.
    """
    daf = DataAccessFacadeBasic()

    rents, appstarts = create_rents_and_appstarts_frames(
        from_date, today, zone_id, cell_ids
    )

    gd = GetHistoricalDemand(rents_df=rents, appstarts_df=appstarts)

    raw_demand: pd.DataFrame = gd.get_demand()
    raw_demand["date"] = raw_demand.time.dt.date
    raw_demand["weekday"] = raw_demand.time.dt.weekday

    distance_cc = daf.find_distance_cell_cell_array(zone_id)

    # 1. group raw demand by weekday,
    # 2. for each weekday group by day
    # 3. for each day blur aggregated demand
    # 4. for each weekday get mean of aggregated demands
    demand_dhc = np.array(
        [
            np.mean(
                blur_demand_history(
                    demand_history_dhc=np.array(
                        [
                            gd.aggregate_demand(d[1], len(cell_ids))
                            for d in w[1].groupby("date")
                        ]
                    ),
                    distance_cc=distance_cc,
                    blur_distance=float(blur_distance),
                    blur_factor=float(blur_factor),
                ),
                axis=0,
            )
            for w in raw_demand.groupby("weekday")
        ]
    )
    output_zone_id = output_zone_id or zone_id
    print(f"saving demand 7d")
    daf.save_demand_prediction_array(output_zone_id, today, demand_dhc)


@huey.task(context=True)
def calculate_revenue_matrix(
    zone_id: str, today: dt.date, output_zone_id: str | None = None, task=None
):
    """
    Create and save revenue estimation matrix(.npy) in filesystem
    """
    log = get_task_logging_adapter(base_logger, task)
    print(f"Begin calculate revenue matrix for zone {zone_id} and date {today}")
    weekday: int = today.weekday()

    daf: DataAccessFacadeBasic = DataAccessFacadeBasic()

    print(f"loading demand prediction")
    demand: np.array = daf.find_demand_prediction_array(
        output_zone_id or zone_id, from_date=today
    )[0]
    print("loading directions")
    directions: np.array = daf.find_directions_array(zone_id, weekday=weekday)
    print("loading distances")
    distances: np.array = daf.find_route_distance_cell_cell_array(zone_id)
    print("loading times")
    times: np.array = daf.find_route_time_cell_cell_array(zone_id)
    print("loading time traffic factor")
    time_traffic_factor: np.array = daf.find_time_traffic_factor_array(zone_id)

    print("applying time traffic factor")
    dhcc = np.zeros((24, *distances.shape))
    thcc = np.zeros((24, *times.shape))
    for i in range(24):
        dhcc[i] = distances
        thcc[i] = times * time_traffic_factor[weekday, i]
        print(f"avg speed for hour {i}: {np.sum(dhcc[i]) / np.sum(thcc[i])}m/s")
    distances = dhcc.astype(np.uint16)
    times = thcc.astype(np.uint16)

    simulator: FastSimulator = FastSimulator()
    print("preparing simulator")
    simulator.prepare(
        demand_hc=demand,
        direction_hcc=directions,
        distance_hcc=distances,
        time_hcc=times,
        simulations_nbr=500,
    )
    print("generating revenue estimation")
    revenue_estimation: RevenueEstimation = generate_revenue_estimation(
        fast_simulator=simulator
    )
    revenue: np.array = revenue_estimation.revenue
    parameters_array: np.array = revenue_estimation.parameters_array

    print("saving revenue and parameters arrays")
    output_zone_id = output_zone_id or zone_id
    daf.save_revenue_array(output_zone_id, from_date=today, revenue_array=revenue)
    daf.save_parameters_array(
        output_zone_id, from_date=today, parameters_array=parameters_array
    )
    print(f"End calculate revenue matrix")


@huey.task(context=True)
def calculate_directions(
    zone_id: str,
    today: dt.date,
    start_date: dt.date | None = None,
    output_zone_id: str | None = None,
    task=None,
) -> None:
    """
    Load:
        - List of rents by zone_id from Internal API.
    Execute:
        Create and save 7 directions matrices(.npy).
    """
    log = get_task_logging_adapter(base_logger, task)
    print(f"Begin calculate directions for zone {zone_id} and date {today}")
    upper_date = dt.datetime(today.year, today.month, today.day, 0).date()
    lower_date = start_date or upper_date - dt.timedelta(weeks=6)

    daf = DataAccessFacadeBasic()
    print("loading zone grid files")
    feats: List[Dict[str, str]] = daf.download_zone_grid_files(zone_id)
    print("creating cell geometries")
    cell_geometries: List[Polygon] = [
        Polygon(f["geometry"]["coordinates"][0]) for f in feats
    ]
    print("loading rents")
    rents: pd.DataFrame = daf.find_rents_frame(
        zone_id, lower_date, upper_date
    ).sort_values(by="start_date")

    gd = GetHistoricalDirections(rents_df=rents, cell_geometries=cell_geometries)
    output_zone_id = output_zone_id or zone_id
    for weekday in range(7):
        print(f"calculating directions for weekday {weekday}")
        d: np.array = gd.get_directions_by_weekday(weekday)
        print(f"saving directions")
        daf.save_directions_array(
            directions_array=d, zone_id=output_zone_id, weekday=weekday
        )
    print("End calculate directions")


@huey.task(context=True)
def calculate_historical_demand_for_date_range(
    zone_id: str,
    cell_ids: list[str],
    start_date: dt.date,
    end_date: dt.date,
    output_zone_id: str | None = None,
    task=None,
) -> None:
    """
    Execute historical demand calculation for given zone and time period.
    """
    log = get_task_logging_adapter(base_logger, task)
    print(
        f"Begin calculating historical demand for zone {zone_id} and date range {start_date} - {end_date}"
    )
    if start_date > end_date:
        log.error("start_date must be smaller than end_date")
        return
    print("downloading science config")
    today = copy(start_date)
    tasks = []
    while today < end_date:
        print(f"calculating demand for date {today}")
        tasks.append(
            calculate_historical_demand(
                today=today,
                zone_id=zone_id,
                cell_ids=cell_ids,
                output_zone_id=output_zone_id,
            )
        )
        today += dt.timedelta(days=1)
    _ = [t.get(blocking=True) for t in tasks]
    print(f"End calculating historical demand for date range")


@huey.task(context=True)
def create_nearest_petrol_station_distance(
    zone_id: str, cell_ids: list[str], start_date: dt.datetime, out_zone: str, task=None
) -> bool:
    """
    Create two vectors of length N with distances and times of shortest routes from each cell to station.
    N is a number of cells in base_zone.
    """
    log = get_task_logging_adapter(base_logger, task)

    daf = DataAccessFacadeBasic()

    log.info("loading gas stations")
    gas_stations: pd.DataFrame = daf.find_gas_stations(zone_id)

    dmc: DistanceTimeMatrixCreator = DistanceTimeMatrixCreator()

    log.info("preparing start and end coords")
    start_coors: np.ndarray = np.array(
        [[float(c) for c in cid.split("-")] for cid in cell_ids]
    )
    end_coords: np.ndarray = gas_stations[["Lon", "Lat"]].values

    log.info("loading cell-station distance and time matrices from HERE api")
    distances_cs, times_cs = dmc.get_distance_and_time_matrices(
        starts_coor=start_coors, destinations_coor=end_coords
    )
    log.info("loading station-cell distance and time matrices from HERE api")
    distances_sc, times_sc = dmc.get_distance_and_time_matrices(
        starts_coor=end_coords, destinations_coor=start_coors
    )

    log.info("flattening matrices to 1d")
    distances = distances_cs + distances_sc.T
    times = times_cs + times_sc.T

    # get indices of closest stations
    min_indices = np.argsort(distances, axis=1)
    nearest_stations = min_indices[:, 0]
    gas_stations["cell_ids"] = [[] for _ in range(len(gas_stations))]
    for i in range(len(nearest_stations)):
        gas_stations["cell_ids"][nearest_stations[i]].append(i)

    # 2d -> 1d
    distances: np.array = np.take_along_axis(distances, min_indices, axis=1)[
        :, 0
    ].astype(np.float64)
    times: np.array = np.take_along_axis(times, min_indices, axis=1)[:, 0].astype(
        np.float64
    )

    assert len(cell_ids) == distances.shape[0]
    assert len(cell_ids) == times.shape[0]

    log.info("saving distance and time matrices")
    daf = DataAccessFacadeBasic()
    daf.save_distance_cell_station_array(out_zone, distances)
    daf.save_time_cell_station_array(out_zone, times)
    log.info("saving gas stations frame")
    daf.save_gas_stations_frame_sd(gas_stations_frame=gas_stations, zone_id=out_zone)

    return True
