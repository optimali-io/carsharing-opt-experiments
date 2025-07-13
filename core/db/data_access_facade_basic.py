import json
import logging
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Dict, List, Union, Tuple

import fiona
import numpy as np
import pandas as pd
from fastkml import KML
from folium import Map
from pandas import DataFrame

from config import settings
from core.db.raw_data_access import dataframe_from_time_period
from core.utils.id_to_index_mapper import IdToIndexMapper
from fleet_manager.data_model.config_models import ScienceConfig
from fleet_manager.data_model.genetic_model import OptimisationResult
from fleet_manager.data_model.zone_model import ServiceTeam
from service_designer.data.data_config import ExperimentConfig
from service_designer.simulator.parallel_simulator import SimulationResult

log = logging.getLogger(__name__)


def create_rents_and_appstarts_frames(
        lower_date: date, upper_date: date, zone_id: str, cell_ids: list[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function creates tuple of rents and appstarts filtered by dates and cell ids.
    """

    cell_id_mapper: IdToIndexMapper = IdToIndexMapper(original_ids=cell_ids)

    daf = DataAccessFacadeBasic()

    rents: pd.DataFrame = daf.find_rents_frame(
        zone_id=zone_id,
        from_date=lower_date,
        to_date=upper_date,
    )

    appstarts: pd.DataFrame = daf.find_user_events_frame(
        zone_id=zone_id,
        from_date=lower_date,
        to_date=upper_date,
    )
    appstarts.event_datetime = pd.to_datetime(appstarts.event_datetime)
    log.info("mapping cell ids for rents and appstarts")
    rents["start_cell_id"] = cell_id_mapper.map_ids_to_indices(
        rents["start_cell_id"].to_list()
    )
    rents["end_cell_id"] = cell_id_mapper.map_ids_to_indices(
        rents["end_cell_id"].to_list()
    )
    appstarts["cell_id"] = cell_id_mapper.map_ids_to_indices(
        appstarts["cell_id"].to_list()
    )

    return rents.sort_values(by="start_date"), appstarts.sort_values(
        by="event_datetime"
    )


class DataAccessFacadeBasic:
    """Data persistence facade basic implementation"""

    def save_simulation_result(self, result: SimulationResult):
        """Save simulation result"""
        pth = f"{settings.SDESIGNER_RESULTS_DIR}/{result.experiment_id}/{result.score}_{len(result.subzone_cell_ids)}"
        Path(pth).mkdir(parents=True, exist_ok=True)
        with open(pth + "/simulation_result.json", "w") as f:
            json.dump(result.model_dump(), f)

    def get_experiment_config_by_id(self, experiment_id: str) -> ExperimentConfig:
        pth = f"{settings.SDESIGNER_DIR}/experiment_configs/experiment_config_{experiment_id}.json"
        with open(pth, "r") as f:
            return ExperimentConfig.model_validate(json.load(f))

    def get_science_config_by_zone_id(self, zone_id: str) -> ScienceConfig:
        with open(f"{settings.ZONES_PATH}/{zone_id}/science_config.json", "r") as f:
            return ScienceConfig.model_validate(json.load(f))

    def load_service_teams(self, zone_id: str) -> List[ServiceTeam]:
        """
        Function creates list of ServiceTeam objects.

        :param: zone_id - ID of the zone for which service teams are loaded.
        :return: - List of ServiceTeam objects.
        """
        service_teams = []
        with open(f"{settings.ZONES_PATH}/{zone_id}/service_teams.json", "r") as f:
            for d in json.load(f):
                service_team_object = ServiceTeam(**d)
                service_teams.append(service_team_object)
        log.info(f"n service teams: {len(service_teams)}")
        assert len(service_teams) > 0
        service_teams = sorted(service_teams, key=lambda st: st.service_team_kind)
        for s in service_teams:
            log.info(s.__dict__)
        return service_teams

    def save_optimisation_result(
            self, result: OptimisationResult, optimisation_datetime: datetime, map: Map
    ):
        """Save optimisation result"""
        pth = f"{settings.RESULTS_DIR}/{result.science_config.zone.id}_{optimisation_datetime.strftime('%Y%m%dT%H%M%S')}"
        Path(pth).mkdir(parents=True, exist_ok=True)
        result.best_phenotype = result.best_phenotype.to_dict()
        with open(pth + "/result.json", "w") as f:
            json.dump(result.model_dump(), f)

        map.save(pth + "/optimisation_map.html")

    def find_rents_frame(
            self, zone_id: str, from_date: date, to_date: date
    ) -> DataFrame:
        pth = f"{settings.DATA_DIR}/rents/{zone_id}/"
        df = dataframe_from_time_period(pth, from_date, to_date)
        df[["start_lon", "start_lat", "end_lon", "end_lat"]] = df[
            ["start_lon", "start_lat", "end_lon", "end_lat"]
        ].apply(pd.to_numeric, errors="coerce")
        df.start_date = pd.to_datetime(df.start_date, errors="coerce")
        df.end_date = pd.to_datetime(df.end_date, errors="coerce")
        return df.dropna()

    def find_user_events_frame(
            self, zone_id: str, from_date: date, to_date: date
    ) -> DataFrame:
        """Find user events frame for zone_id and date range"""
        pth = f"{settings.DATA_DIR}/app_events/{zone_id}/"
        df = dataframe_from_time_period(pth, from_date, to_date)
        df[["lon", "lat"]] = df[["lon", "lat"]].apply(pd.to_numeric, errors="coerce")
        df.event_datetime = pd.to_datetime(df.event_datetime, errors="coerce")

        return df[df.event_datetime.notnull()]

    def find_demand_history_array(
            self, zone_id: str, from_date: date, to_date: date
    ) -> np.ndarray:
        """
        Download and join demand matrices from given time range. Returned matrix will be of size n_daysx24xn_cells.
        :param zone_id: Zone id
        :param from_date: dt.date, start date of a demand time range
        :param to_date: dt.date, end date of a demand time range
        """
        demand_history_dhc = None
        days = (to_date - from_date).days + 1
        for actual_day_idx in range(days):
            actual_date = from_date + timedelta(days=actual_day_idx)
            log.info(f"load demand for date {actual_date}")
            with open(
                    f"{settings.ZONES_PATH}/{zone_id}/demand_history_{actual_date}.npy",
                    "rb",
            ) as f:
                demand_history_hc: np.ndarray = np.load(f)
            hours, cells = demand_history_hc.shape
            if demand_history_dhc is None:
                demand_history_dhc = np.zeros(
                    (days, hours, cells), dtype=demand_history_hc.dtype
                )
            demand_history_dhc[actual_day_idx] = demand_history_hc
        return demand_history_dhc

    def save_demand_history_array(
            self, zone_id: str, from_date: date, demand_history_hc: np.ndarray
    ):
        """
        Save provided demand matrix.
        :param zone_id: Zone id
        :param from_date: dt.date, day of a demand
        :param demand_history_hc: np.ndarray, demand matrix to be saved
        """
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        fname = f"/demand_history_{from_date}.npy"
        Path(pth).mkdir(parents=True, exist_ok=True)
        with open(pth + fname, "wb") as f:
            np.save(f, demand_history_hc)

    def find_demand_prediction_array(self, zone_id: str, from_date: date) -> np.ndarray:
        """
        Download demand prediction matrix from S3.
        :param zone_id: Zone id
        :param from_date: dt.date, date of the demand prediction
        """
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/demand_prediction_{from_date}.npy", "rb"
        ) as f:
            demand_prediction_dhc: np.ndarray = np.load(f)
            return demand_prediction_dhc

    def save_demand_prediction_array(
            self, zone_id: str, from_date: date, demand_prediction_dhc: np.ndarray
    ) -> str:
        """
        Save provided demand prediction matrix.
        :param zone_id: Zone id
        :param from_date: dt.date, date of the demand prediction
        :param demand_prediction_dhc: np.ndarray, demand prediction matrix to be saved
        """
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        fname = f"/demand_prediction_{from_date}.npy"
        Path(pth).mkdir(parents=True, exist_ok=True)
        with open(pth + fname, "wb") as f:
            np.save(f, demand_prediction_dhc)

    def download_zone_grid_files(self, zone_id: str) -> List[Dict]:
        """Download shapefiles if necessary, save them locally and return list of features in geojson format."""
        local_path = f"{settings.ZONES_PATH}/{zone_id}/zone_grid.shp"
        with fiona.open(local_path) as shp:
            # Shp features must always be sorted by property id in ascending order to match indices in result matrix.
            feats: List[Dict[str, str]] = sorted(
                list(shp), key=lambda d: d["properties"]["id"], reverse=False
            )
        return feats

    def download_zone_kml(self, zone_param: Union[str, Dict]):
        """Download zone.kml if necessary and return an KML object"""
        local_path = f"{settings.ZONES_PATH}/{zone_param}/zone.kml"
        with open(local_path, "r", encoding="utf8", errors="ignore") as f:
            contents = f.read()
        kml = KML()
        kml.from_string(contents)
        return kml

    def save_directions_array(
            self, directions_array: np.array, zone_id: str, weekday: int
    ):
        """Save directions array locally and upload it to S3."""
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        fname = f"/directions_{weekday}.npy"
        Path(pth).mkdir(parents=True, exist_ok=True)
        with open(pth + fname, "wb") as f:
            np.save(f, directions_array)

    def find_directions_array(self, zone_id: str, weekday: int) -> np.ndarray:
        """Load directions array from filesystem or from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/directions_{weekday}.npy", "rb"
        ) as f:
            directions_array: np.ndarray = np.load(f)
            return directions_array

    def save_time_traffic_factor_array(
            self, zone_id: str, time_traffic_factor_array: np.ndarray
    ):
        """Save time traffic factor array locally and upload it to S3."""
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        fname = f"/time_traffic_factor.npy"
        Path(pth).mkdir(parents=True, exist_ok=True)
        with open(pth + fname, "wb") as f:
            np.save(f, time_traffic_factor_array)

    def find_time_traffic_factor_array(self, zone_id: str) -> np.ndarray:
        """Load time traffic factor array from filesystem or load it from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/time_traffic_factor.npy", "rb"
        ) as f:
            time_traffic_factor_array: np.ndarray = np.load(f)
            return time_traffic_factor_array

    def find_distance_cell_cell_array(self, zone_param: Union[str, Dict]) -> np.ndarray:
        """Load distance cell-cell array from filesystem or from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_param}/distance_cell_cell.npy", "rb"
        ) as f:
            distance_cell_cell_array: np.ndarray = np.load(f)
            return distance_cell_cell_array

    def find_time_cell_cell_array(self, zone_param: Union[str, Dict]) -> np.ndarray:
        """Load time cell-cell array from filesystem or from S3."""
        with open(f"{settings.ZONES_PATH}/{zone_param}/time_cell_cell.npy", "rb") as f:
            time_cell_cell_array: np.ndarray = np.load(f)
            return time_cell_cell_array

    def find_distance_cell_station_cell_array(self, zone_id: str) -> np.ndarray:
        """Load distance cell-station-cell array from filesystem or from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/distance_cell_station_cell.npy", "rb"
        ) as f:
            distance_cell_cell_array: np.ndarray = np.load(f)
            return distance_cell_cell_array

    def find_time_cell_station_cell_array(self, zone_id: str) -> np.ndarray:
        """Load time cell-station-cell array from filesystem or from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/time_cell_station_cell.npy", "rb"
        ) as f:
            return np.load(f)

    def find_id_cell_station_cell_array(self, zone_id: str) -> np.ndarray:
        """Load id cell-station-cell array from filesystem or from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/id_cell_station_cell.npy", "rb"
        ) as f:
            return np.load(f)

    def save_route_distance_cell_cell_array(
            self, zone_id: str, route_distance_cell_cell_array: np.ndarray
    ):
        """Save route distance cell cell array locally and upload it to S3."""
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        fname = f"/route_distance_cell_cell.npy"
        Path(pth).mkdir(parents=True, exist_ok=True)
        with open(pth + fname, "wb") as f:
            np.save(f, route_distance_cell_cell_array)

    def save_route_time_cell_cell_array(
            self, zone_id: str, route_time_cell_cell_array: np.ndarray
    ):
        """Save route time cell cell array locally and upload it to S3."""
        fname = f"/route_time_cell_cell.npy"
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        Path(pth).mkdir(parents=True, exist_ok=True)
        with open(pth + fname, "wb") as f:
            np.save(f, route_time_cell_cell_array)

    def find_route_distance_cell_cell_array(self, zone_id: str) -> np.ndarray:
        """Load route distance cell-cell array from filesystem or from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/route_distance_cell_cell.npy", "rb"
        ) as f:
            route_distance_cell_cell_array: np.ndarray = np.load(f)
            return route_distance_cell_cell_array

    def find_route_time_cell_cell_array(self, zone_id: str) -> np.ndarray:
        """Load route time cell-cell array from filesystem or from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/route_time_cell_cell.npy", "rb"
        ) as f:
            route_time_cell_cell_array: np.ndarray = np.load(f)
            return route_time_cell_cell_array

    def save_revenue_array(
            self, zone_id: str, from_date: date, revenue_array: np.ndarray
    ):
        """Save revenue estimation array locally and upload it to S3."""
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        Path(pth).mkdir(parents=True, exist_ok=True)
        fname = f"/revenue_array_{from_date}.npy"
        with open(pth + fname, "wb") as f:
            np.save(f, revenue_array)

    def find_revenue_array(self, zone_id: str, from_date: date) -> np.ndarray:
        """Load revenue estimation array from filesystem or load it from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/revenue_array_{from_date}.npy", "rb"
        ) as f:
            return np.load(f)

    def save_parameters_array(
            self, zone_id: str, from_date: date, parameters_array: np.ndarray
    ):
        """Save revenue estimation array locally and upload it to S3."""
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        Path(pth).mkdir(parents=True, exist_ok=True)
        fname = f"/revenue_parameters_{from_date}.npy"
        with open(pth + fname, "wb") as f:
            np.save(f, parameters_array)

    def find_parameters_array(self, zone_id: str, from_date: date) -> np.ndarray:
        """Load revenue estimation array from filesystem or load it from S3."""
        with open(
                f"{settings.ZONES_PATH}/{zone_id}/revenue_parameters_{from_date}.npy", "rb"
        ) as f:
            return np.load(f)

    def find_gas_stations(self, zone_param: Union[str, Dict]) -> pd.DataFrame:
        """Load gas stations from filesystem or load it from S3."""
        with open(f"{settings.ZONES_PATH}/{zone_param}/petrol_stations.csv", "r") as f:
            return pd.read_csv(f, sep=";")

    def find_power_stations(self, zone_param: Union[str, Dict]) -> pd.DataFrame:
        """Load charge stations from filesystem or load it from S3."""
        with open(f"{settings.ZONES_PATH}/{zone_param}/power_stations.csv", "r") as f:
            return pd.read_csv(f, sep=";")

    def save_distance_cell_station_array(self, zone_id: str, distance_cell_station_array: np.ndarray):
        """Save distance cell-station array locally and upload it to S3."""
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        fname = f"/distance_cell_station.npy"
        Path(pth).mkdir(parents=True, exist_ok=True)
        with open(pth + fname, "wb") as f:
            np.save(f, distance_cell_station_array)

    def save_time_cell_station_array(self, zone_id: str, time_cell_station_array: np.ndarray):
        """Save time cell-station array locally and upload it to S3."""
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        fname = f"/time_cell_station.npy"
        Path(pth).mkdir(parents=True, exist_ok=True)
        with open(pth + fname, "wb") as f:
            np.save(f, time_cell_station_array)

    def find_distance_cell_station_array(self, zone_id: str) -> np.ndarray:
        """Load distance cell-station array from filesystem"""
        with open(f"{settings.ZONES_PATH}/{zone_id}/distance_cell_station.npy", "rb") as f:
            return np.load(f)

    def find_time_cell_station_array(self, zone_id: str) -> np.ndarray:
        """Load time cell-station array from filesystem"""
        with open(f"{settings.ZONES_PATH}/{zone_id}/time_cell_station.npy", "rb") as f:
            return np.load(f)

    def save_gas_stations_frame_sd(self, zone_id: str, gas_stations_frame: pd.DataFrame):
        """Save gas stations frame to CSV."""
        pth = f"{settings.ZONES_PATH}/{zone_id}"
        fname = f"/petrol_stations_sd.csv"
        Path(pth).mkdir(parents=True, exist_ok=True)
        gas_stations_frame.to_csv(pth + fname, sep=";", index=False)

    def find_gas_stations_frame_sd(self, zone_id: str):
        pth = f"{settings.ZONES_PATH}/{zone_id}/petrol_stations_sd.csv"
        return pd.read_csv(pth, sep=";")
