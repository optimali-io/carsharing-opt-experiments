import datetime as dt
import logging
import os.path
import pickle
import random
from itertools import chain
from multiprocessing import Pool
from time import sleep
from typing import Dict, List, Union

import numpy as np
from fastkml import KML
from folium import Map
from huey.api import Result

from config import huey, get_task_logging_adapter

from core.db.data_access_facade_basic import DataAccessFacadeBasic
from core.db.raw_data_access import load_vehicles_from_zone
from fleet_manager.data_model.genetic_model import Phenotype, OptimisationResult
from fleet_manager.data_model.config_models import GeneticConfiguration, ScienceConfig
from fleet_manager.data_model.zone_model import ZoneData
from fleet_manager.genetic_algorithm.genetic_controller_parallel import (
    GeneticControllerParallelDynamicIslands,
)
from fleet_manager.genetic_algorithm.genetic_worker import (
    run_genetic_worker,
)
from fleet_manager.genetic_algorithm.runners import IGeneticWorkersRunner
from fleet_manager.genetic_output_builder.map_plotter import MapPlotter
from fleet_manager.tasks.build_stations import build_gas_stations, build_power_stations

base_logger = logging.getLogger("fleet_manager")


@huey.task(context=True)
def load_zone_data(
    science_config: ScienceConfig,
    optimisation_datetime: dt.datetime,
    task=None,
    finished: bool | None = None,
) -> tuple[ScienceConfig, ZoneData]:
    """
    Load:
        - Demand prediction matrix(.npy) from Redis by zone_id for given hour
        - Revenue estimation matrix(.npy) from Redis by zone_id for given hour
        - Number of city and zone cells from Dict
        - List of gas stations from Backend DB through Internal API
        - List of service teams from Backend DB through Internal API
        - List of vehicles from Backend DB through Internal API
        - Distances, times, distances through petrol station and times through petrol station matrices(.npy) from S3
    Execute:
        Create and initialize ZoneData object.
    Return:
        ZoneData object
    """
    zone_id: str = science_config.zone.id
    log = get_task_logging_adapter(base_logger, task)
    log.info(f"Begin loading ZoneData for zone: {zone_id}")

    zone_data: ZoneData = ZoneData.from_science_config(science_config)
    daf = DataAccessFacadeBasic()

    from_date = optimisation_datetime.date()

    log.info("loading demand")
    zone_data.demand = np.sum(
        daf.find_demand_prediction_array(zone_id=zone_id, from_date=from_date)[
            from_date.weekday()
        ],
        0,
    )
    log.info(f"demand sum: {np.sum(zone_data.demand)}")
    log.info("loading revenue")
    zone_data.revenue = daf.find_revenue_array(zone_id=zone_id, from_date=from_date)
    log.info(f"revenue sum: {np.sum(zone_data.revenue)}")
    zone_data.cell_number = len(science_config.zone.cell_ids)
    log.info(f"cell number: {zone_data.cell_number}")
    zone_data.zone_cell = list(range(zone_data.cell_number))
    log.info("loading distances")
    zone_data.distance_cell_cell = daf.find_distance_cell_cell_array(zone_id)
    log.info(f"distance sum: {np.sum(zone_data.distance_cell_cell)}")
    log.info("loading times")
    zone_data.time_cell_cell = daf.find_time_cell_cell_array(zone_id)
    log.info(f"time sum: {np.sum(zone_data.time_cell_cell)}")
    log.info("loading station distances")
    zone_data.distance_cell_station_cell = daf.find_distance_cell_station_cell_array(
        zone_id
    )
    log.info(f"distance station sum: {np.sum(zone_data.distance_cell_station_cell)}")
    log.info("loading station times")
    zone_data.time_cell_station_cell = daf.find_time_cell_station_cell_array(zone_id)
    log.info(f"time station sum: {np.sum(zone_data.distance_cell_cell)}")
    log.info("loading station ids")
    zone_data.id_cell_station_cell = daf.find_id_cell_station_cell_array(zone_id)

    # service teams
    daf = DataAccessFacadeBasic()
    zone_data.service_team = daf.load_service_teams(zone_id)

    # fleet
    zone_data.fleet = load_vehicles_from_zone(science_config, optimisation_datetime)

    zone_data.init_zone_data()
    zone_data.gas_station = build_gas_stations(
        daf.find_gas_stations(zone_id), zone_data
    )

    log.info("initializing zone data")
    log.info(f"n vehicles to refuel: {len(zone_data.fleet_ids_to_refuel)}")
    log.info(
        f"vehicles to refuel ranges: {[zone_data.fleet[idx].range for idx in zone_data.fleet_ids_to_refuel]}"
    )
    log.info(f"End loading ZoneData")

    return science_config, zone_data


@huey.task(context=True)
def load_zone_data_electric_mock(
    science_config: Dict, date: dt.date, shifts_ids: List[int], task=None
) -> ZoneData:
    """
    Load:
        - Demand prediction matrix(.npy) from Redis by zone_id for given hour
        - Revenue estimation matrix(.npy) from Redis by zone_id for given hour
        - Number of city and zone cells from Dict
        - List of gas stations from Backend DB through Internal API
        - List of service teams from Backend DB through Internal API
        - List of vehicles from Backend DB through Internal API
        - Distances, times, distances through petrol station and times through petrol station matrices(.npy) from S3
    Execute:
        Create and initialize ZoneData object.
    Return:
        ZoneData object
    """
    zone_id: str = science_config["zone"]["id"]
    log = get_task_logging_adapter(base_logger, task)
    log.info(f"Begin loading ZoneData for zone: {zone_id}")

    zone_data: ZoneData = ZoneData(science_config)
    daf = DataAccessFacadeBasic()

    log.info("loading demand")
    zone_data.demand = np.sum(
        daf.find_demand_prediction_array(zone_id=zone_id, from_date=date)[
            date.weekday()
        ],
        0,
    )
    log.info(f"demand sum: {np.sum(zone_data.demand)}")
    log.info("loading revenue")
    zone_data.revenue = daf.find_revenue_array(zone_id=zone_id, from_date=date)
    log.info(f"revenue sum: {np.sum(zone_data.revenue)}")
    zone_data.cell_number = len(science_config["zone"]["cell_ids"])
    log.info(f"cell number: {zone_data.cell_number}")
    zone_data.zone_cell = list(range(zone_data.cell_number))
    log.info("loading distances")
    zone_data.distance_cell_cell = daf.find_distance_cell_cell_array(zone_id)
    log.info(f"distance sum: {np.sum(zone_data.distance_cell_cell)}")
    log.info("loading times")
    zone_data.time_cell_cell = daf.find_time_cell_cell_array(zone_id)
    log.info(f"time sum: {np.sum(zone_data.time_cell_cell)}")
    log.info("loading station distances")
    zone_data.distance_cell_station_cell = daf.find_distance_cell_station_cell_array(
        zone_id
    )
    log.info(f"distance station sum: {np.sum(zone_data.distance_cell_station_cell)}")
    log.info("loading station times")
    zone_data.time_cell_station_cell = daf.find_time_cell_station_cell_array(zone_id)
    log.info(f"time station sum: {np.sum(zone_data.distance_cell_cell)}")
    log.info("loading station ids")
    zone_data.id_cell_station_cell = daf.find_id_cell_station_cell_array(zone_id)

    zone_data.fleet = load_vehicles_from_zone(science_config)
    zone_data.initialize_cells()
    zone_data.power_station = build_power_stations(
        daf.find_power_stations(zone_id), zone_data
    )

    for ps in zone_data.power_station:
        ps.base_charging_vehicles = []
    fleet_size = len(zone_data.fleet)
    ids_selected_for_electric = np.random.choice(
        list(range(fleet_size)), int(0.75 * fleet_size)
    )
    low_range_electric_ids = np.random.choice(
        ids_selected_for_electric, int(0.5 * len(ids_selected_for_electric))
    )
    low_range_gas = np.random.choice(
        [
            vid
            for vid in list(range(fleet_size))
            if vid not in ids_selected_for_electric
        ],
        int(0.5 * (fleet_size - len(ids_selected_for_electric))),
    )
    plugged_vehicles_ids = np.random.choice(
        [i for i in ids_selected_for_electric if i not in low_range_electric_ids],
        int(0.2 * len(ids_selected_for_electric)),
    )
    for i in ids_selected_for_electric:
        zone_data.fleet[i].drive = 2
    for i in low_range_electric_ids:
        zone_data.fleet[i].range = 70
    for i in plugged_vehicles_ids:
        power_station = random.choice(zone_data.power_station)
        zone_data.fleet[i].power_station_id = power_station.power_station_id
        zone_data.fleet[i].latitude = power_station.latitude
        zone_data.fleet[i].longitude = power_station.longitude

    for i in low_range_gas:
        zone_data.fleet[i].range = 70

    daf = DataAccessFacadeBasic()
    zone_data.service_team = daf.load_service_teams(zone_id)
    zone_data.cells = []
    zone_data.init_zone_data()
    zone_data.gas_station = build_gas_stations(
        daf.find_gas_stations(zone_id), zone_data
    )

    log.info("initializing zone data")
    log.info(f"n vehicles to refuel: {len(zone_data.fleet_ids_to_refuel)}")
    log.info(
        f"vehicles to refuel ranges: {[zone_data.fleet[idx].range for idx in zone_data.fleet_ids_to_refuel]}"
    )
    log.info(f"End loading ZoneData")
    return zone_data


@huey.task(context=True)
def run_genetic_algorithm(
    science_config: ScienceConfig,
    zone_data: ZoneData,
    optimisation_datetime: dt.datetime,
    task=None,
) -> OptimisationResult:
    gconf = science_config.genetic_config
    log = get_task_logging_adapter(base_logger, task)
    log.info("create workers runner")
    worker_runner = HueyGeneticWorkersRunner(gconf, zone_data)
    log.info("create genetic controller")
    gc = GeneticControllerParallelDynamicIslands(gconf, zone_data)
    gc.set_workers_runner(worker_runner)
    log.info("start genetic algorithm execution")

    log.info(
        f"{dt.datetime.utcnow()} Start executing GeneticControllerParallelDynamic."
    )
    time_start = dt.datetime.now()
    gc.execute()
    time_end = dt.datetime.now()
    optimization_time: float = (time_end - time_start).total_seconds()
    log.info(f"GeneticControllerParallelDynamic execution time: {optimization_time}")

    best_phenotype = gc.best_phenotype
    remove_data_from_redis(worker_runner.zone_data_id)
    log.info(f"best phenotype: {best_phenotype}")
    log.info("End running genetic algorithm")
    daf = DataAccessFacadeBasic()

    result = OptimisationResult(
        best_phenotype=best_phenotype,
        optimization_time=optimization_time,
        science_config=science_config,
    )
    m = create_map(best_phenotype, zone_data, science_config)
    daf.save_optimisation_result(result, optimisation_datetime, m)
    return result


def store_data_in_redis(key: str, value: Union[str, bytes]):
    """
    Function stores given data in Redis
    :param key: key for Redis
    :param value: string or bytes data to store
    """
    huey.put(key, value)


def remove_data_from_redis(key: str):
    """
    Function removes data from Redis by key
    :param key: key for Redis
    """
    huey.get(key)


@huey.task(context=True)
def run_genetic_worker_task(
    population: List[Phenotype],
    configuration: GeneticConfiguration,
    zone_data_id: str,
    task=None,
):
    """
    Task loads serialized the ZoneData instance from Redis or filesystem and calls a genetic worker runner.
    :param population: list of a Phenotype instances (previous generation)
    :param configuration: GeneticConfiguration instance.
    :param zone_data_id: timestamp of a given ZoneData instance and also the key from Redis storage
    :return: new generation (list) of Phenotype instances (new generation)
    """
    log = get_task_logging_adapter(base_logger, task)
    zone_data_path: str = f"/tmp/{zone_data_id}_{os.getpid()}"
    if not os.path.exists(zone_data_path):
        serialized_zone_data: bytes = huey.get(zone_data_id, peek=True)
        zone_data: ZoneData = pickle.loads(serialized_zone_data)
        log.info(
            f"in worker task: no ZoneData file, creating pickle for process {os.getpid()}"
        )
        with open(zone_data_path, "wb") as f:
            pickle.dump(zone_data, f)
    return run_genetic_worker(population, configuration, zone_data_path)


def create_map(
    phenotype: Phenotype, zone_data: ZoneData, science_config: ScienceConfig
) -> Map:
    """
    Create relocation map and return its path.
    :param phenotype: phenotype
    :param zone_data: ZoneData
    :param science_config: science config
    :return: Map instance
    """

    cell_ids = science_config.zone.cell_ids
    zone_id = science_config.zone.id
    daf = DataAccessFacadeBasic()
    kml_file: KML = daf.download_zone_kml(zone_id)

    mp = MapPlotter(zone_data, kml_file, phenotype.to_dict(), cell_ids)
    mp.run_plotting()

    return mp.folium_map


class CpuGeneticWorkersRunner(IGeneticWorkersRunner):
    """
    Class responsible for creation and control of single epoch data_preparation. It will be used instead of a Huey version
    if Redis will be unavailable or not necessary (unit tests, experiments)
    """

    def __init__(
        self,
        configuration: GeneticConfiguration,
        zone_data: ZoneData,
        zone_data_path: str,
    ):
        self._configuration: GeneticConfiguration = configuration
        self._zone_data_path: str = zone_data_path
        with open(zone_data_path, "wb") as f:
            pickle.dump(zone_data, f)

    def run_workers(self, clusters: list[list[Phenotype]]) -> list[Phenotype]:
        """
        Execute worker runner on each cluster.

        :param clusters: list of lists (subpopulations) of the Phenotype instances.

        :return: flattened list of Phenotypes from all processed clusters
        """

        with Pool() as p:
            return list(
                chain.from_iterable(
                    p.starmap(
                        run_genetic_worker,
                        [
                            (c, self._configuration, self._zone_data_path)
                            for c in clusters
                        ],
                    )
                )
            )


class HueyGeneticWorkersRunner(IGeneticWorkersRunner):
    """
    Class responsible for creation and control of single epoch data_preparation.
    """

    def __init__(self, configuration: GeneticConfiguration, zone_data: ZoneData):
        self._configuration: GeneticConfiguration = configuration
        self._zone_data: ZoneData = zone_data
        self.zone_data_id: str = str(dt.datetime.now().timestamp())
        self._store_zone_data()

    def _store_zone_data(self):
        """Serialize ZoneData instance and put it in Redis"""
        base_logger.info(f"serializing zone data")
        serialized_zone_data = pickle.dumps(self._zone_data)
        base_logger.info(f"storing zone data in redis")
        store_data_in_redis(self.zone_data_id, serialized_zone_data)

    def run_workers(self, clusters: list[list[Phenotype]]) -> list[Phenotype]:
        """
        Create as many data_preparation as a number of clusters, execute them and get results.
        :param clusters: list of lists (subpopulations) of the Phenotype instances.
        :return: flattened list of Phenotypes from all processed clusters
        """
        pending: list[Union[Result, None]] = []
        new_population: list[Phenotype] = []
        population_size = 0
        base_logger.info(f"running workers for {len(clusters)} clusters")

        for cluster in clusters:
            pending.append(
                run_genetic_worker_task(cluster, self._configuration, self.zone_data_id)
            )
            population_size += len(cluster)

        while len(new_population) < population_size:
            for i in range(len(pending)):
                if pending[i]:
                    sub_pop = pending[i]()
                    if sub_pop:
                        new_population.extend(sub_pop)
                        pending[i] = None
                else:
                    continue
            sleep(0.1)
        return new_population
