import logging
from typing import List

import pandas as pd

from fleet_manager.data_model.zone_model import GasStation, PowerStation, ZoneData

log = logging.getLogger("fleet_managers")


def build_gas_stations(gas_stations_df: pd.DataFrame, zone_data: ZoneData) -> List[GasStation]:
    """
    Creates list of GasStation objects from gas_stations DataFrame.

    :param: gas_stations_df - gas stations
    :return: - List of GasStation objects
    """
    gas_stations: List[GasStation] = []
    for idx, row in enumerate(gas_stations_df.values):
        cell_id: int = zone_data.get_nearest_cell_id_for_point(row[0], row[1])
        gs = GasStation(
            gas_station_id=idx,
            gas_station_name=row[2],
            latitude=row[0],
            longitude=row[1],
            cell_id=cell_id,
            address=row[3],
        )
        gas_stations.append(gs)
    log.info(f"{len(gas_stations)} built")
    return gas_stations


def build_power_stations(power_stations_df: pd.DataFrame, zone_data: ZoneData) -> List[PowerStation]:
    """
    Creates list of ChargeStation objects from power_stations DataFrame.

    :param: power_stations_df - power stations
    :return: - List of ChargeStation objects
    """
    power_stations: List[PowerStation] = []
    for idx, row in enumerate(power_stations_df.itertuples()):
        cell_id: int = zone_data.get_nearest_cell_id_for_point(row.latitude, row.longitude)
        cs = PowerStation(
            power_station_id=idx,
            cell_id=cell_id,
            capacity=row.capacity,
            latitude=row.latitude,
            longitude=row.longitude,
        )
        power_stations.append(cs)
    log.info(f"{len(power_stations)} built")
    return power_stations
