import datetime as dt
import glob
import json
import logging
import os
import re
from datetime import date, datetime
from typing import List

import pandas as pd
from rtree.index import Index
from shapely.geometry import Polygon, Point, shape

from config import settings
from core.utils.id_to_index_mapper import IdToIndexMapper
from fleet_manager.data_model.config_models import ScienceConfig
from fleet_manager.data_model.zone_model import (
    Vehicle,
    VehicleModels,
    VehicleModelName,
    FuelType,
)

log = logging.getLogger("fleet_manager")


def dataframe_from_time_period(dir_path: str, start_date: date, end_date: date):
    if not os.path.exists(dir_path):
        return None
    rents_paths = [
        os.path.join(dir_path, path)
        for path in os.listdir(dir_path)
        if path.endswith(".csv")
    ]
    paths_in_period = _filter_files_by_time_period(
        rents_paths, start_date, end_date
    )
    if paths_in_period:
        return _dataframe_from_files_list(paths_in_period)
    else:
        return None


def _filter_files_by_time_period(
    files_list: list[str], start_date: date, end_date: date
) -> list[str]:
    files_in_period = []
    for f in files_list:
        date_string = re.search("[0-9]{4}-[0-9]{2}-[0-9]{2}", f)[0]
        file_date = datetime.strptime(date_string, "%Y-%m-%d").date()
        if start_date <= file_date <= end_date:
            files_in_period.append(f)
    return files_in_period


def _dataframe_from_files_list(files_list: list[str]) -> pd.DataFrame:
    df_list = []
    for path in files_list:
        df = pd.read_csv(path, sep=";")
        df_list.append(df)
    result_df = pd.concat(df_list)
    return result_df


def get_cell_id_by_coords(lon: float, lat: float, index: Index) -> str:
    """
    Function returns id of cell that contains point of given coordinates or None if cell id can't be determined.

    :param lon - longitude of point
    :param lat - latitude of point
    :param index - spatial index of operator cells
    """
    p = Point(lon, lat)
    candidates = index.intersection((lon, lat), objects="raw")
    if not candidates:
        raise Exception(
            f"Could not assign cell id to position: {p}. No intersection with cells from grid."
        )
    cell_id = None
    for feat in candidates:
        c_geom: Polygon = shape(feat["geometry"])
        if c_geom.intersection(p):
            cell_id = feat["properties"]["id"]
            break
    return cell_id


def load_vehicles_frame(
    data_dir: str,
    start_date_time: dt.datetime,
) -> pd.DataFrame:
    spatial_index = Index(settings.SPATIAL_INDEX_PATH)
    date_time_string = start_date_time.strftime("%Y-%m-%dT%H-%M-%S")

    file_path = glob.glob(f"{data_dir}/*{date_time_string}*")[0]
    with open(file_path, "r") as f:
        vehicles = pd.read_csv(f, sep=";")
    columns = ["obj_id", "status", "lon", "lat", "fuel", "date_time"]
    vehicles = vehicles[columns]
    vehicles.columns = ["id", "status", "lon", "lat", "fuel", "date_time"]
    vehicles.date_time = pd.to_datetime(vehicles.date_time, errors="coerce")
    vehicles["wait_time_minutes"] = (
        start_date_time.replace(tzinfo=dt.UTC) - vehicles["date_time"]
    ).dt.seconds / 60
    vehicles[["lon", "lat"]] = vehicles[["lon", "lat"]].apply(
        pd.to_numeric, errors="coerce"
    )
    cell_ids = [
        get_cell_id_by_coords(lon=row.lon, lat=row.lat, index=spatial_index)
        for row in vehicles.itertuples(index=False)
    ]
    vehicles["cell_id"] = cell_ids
    vehicles["model"] = VehicleModelName.yaris
    return vehicles


def load_vehicles_from_zone(
    science_config: ScienceConfig, start_date_time: dt.datetime
) -> List[Vehicle]:
    """
    Function loads vehicle data and creates list of Vehicle objects.
    """

    def _calculate_range(fuel_level: float, fuel_consumption: float) -> int:
        """
        Calculate range of a vehicle
        :param fuel_level: float, fuel level in litres
        :param fuel_consumption: float, fuel consumption in liters (liters / 100 km)
        :return: float, range in kilometers
        """
        return int(fuel_level * 100 / fuel_consumption)  # km

    log.info("loading fleet")
    fleet: List[Vehicle] = []
    cell_id_to_index_mapper: IdToIndexMapper = IdToIndexMapper(
        original_ids=science_config.zone.cell_ids
    )
    vehicles = load_vehicles_frame(
        data_dir=settings.DATA_DIR + f"/vehicles/{science_config.zone.id}/",
        start_date_time=start_date_time,
    )
    vehicles = vehicles.loc[vehicles["status"] == "available"]

    with open(
        f"{settings.ZONES_PATH}/{science_config.zone.id}/vehicle_models.json", "r"
    ) as f:
        vehicle_models = VehicleModels.model_validate(json.load(f))

    for i, v in enumerate(vehicles.itertuples()):
        lon, lat = v.lon, v.lat
        model = vehicle_models.models[v.model]
        vehicle = Vehicle(
            vehicle_index=i,
            backend_vehicle_id=v.id,
            cell_id=cell_id_to_index_mapper.map_id_to_index(v.cell_id),
            range=_calculate_range(
                fuel_level=float(v.fuel),
                fuel_consumption=float(model.average_fuel_consumption),
            ),
            max_range=_calculate_range(
                fuel_level=float(model.tank_capacity),
                fuel_consumption=float(model.average_fuel_consumption),
            ),
            wait_time=int(v.wait_time_minutes),
            latitude=float(lat),
            longitude=float(lon),
            drive=2 if model.fuel_type == FuelType.ELECTRIC else 1,
        )
        fleet.append(vehicle)
    log.info(f"fleet size: {len(fleet)}")
    assert len(fleet) > 0
    return fleet
