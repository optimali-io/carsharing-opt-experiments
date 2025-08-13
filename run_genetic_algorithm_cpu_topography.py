import datetime as dt
from datetime import timedelta
from enum import Enum
from time import time

from config import huey
from core.db.data_access_facade_basic import DataAccessFacadeBasic
from core.tasks.data_preparation import calculate_demand_prediction, calculate_historical_demand_for_date_range, \
    calculate_directions, calculate_revenue_matrix
from core.tasks.distance_time_modification import run_distance_modification_approximation, run_traffic_factors_task
from fleet_manager.data_model.config_models import ScienceConfig
from fleet_manager.data_model.genetic_model import OptimisationResult
from fleet_manager.genetic_algorithm.genetic_controller_parallel import GeneticControllerParallelCpu
from fleet_manager.tasks.optimisation import load_zone_data, create_map

date_format = "%Y-%m-%d"
date_time_format = "%Y-%m-%d %H:%M:%S"

class TopographyType(str, Enum):
    ISLAND_ONE_WAY_RING = 'IslandOneWayRing'
    ISLAND_TWO_WAY_RING = 'IslandTwoWayRing'
    ISLAND_FULLY_CONNECTED = 'IslandFullyConnected'
    ISLAND_STAR = 'IslandStar'
    ISLAND_LATTICE = 'IslandLattice'
    CELLULAR_LINEAR_FIXED_BORDER = 'CellularLinearFixedBorder4'
    CELLULAR_LINEAR_NO_BORDER = 'CellularLinearNoBorder4'
    CELLULAR_COMPACT_FIXED_BORDER = 'CellularCompactFixedBorder4'
    CELLULAR_COMPACT_NO_BORDER = 'CellularCompactNoBorder4'
    CELLULAR_DIAMOND_FIXED_BORDER = 'CellularDiamondFixedBorder4'
    CELLULAR_DIAMOND_NO_BORDER = 'CellularDiamondNoBorder4'


def optimise_zone_with_data_preparation(zone_id: str, date_time_string: str, topography: str = "IslandOneWayRing"):
    """Run optimisation with data preparation for given zone and date."""
    date_time: dt.datetime = dt.datetime.strptime(date_time_string, date_time_format)
    date: dt.date = date_time.date()
    daf = DataAccessFacadeBasic()
    science_config: ScienceConfig = daf.get_science_config_by_zone_id(zone_id=zone_id)
    science_config.genetic_config.topography_type = topography
    cell_ids: list[str] = science_config.zone.cell_ids
    blur_factor = science_config.zone_data_config.blur_factor
    blur_distance = science_config.zone_data_config.blur_distance
    pipeline = (
        calculate_historical_demand_for_date_range.s(zone_id=zone_id, cell_ids=cell_ids, start_date=date - timedelta(days=28), end_date=date)
        .then(calculate_demand_prediction, zone_id=zone_id, today=date, blur_factor=blur_factor, blur_distance=blur_distance)
        .then(calculate_directions, zone_id=zone_id, today=date)
        .then(run_distance_modification_approximation, zone_id=zone_id, cell_ids=cell_ids,today=date)
        .then(run_traffic_factors_task, zone_id=zone_id, today=date)
        .then(calculate_revenue_matrix, zone_id=zone_id, today=date)
    )
    result_group = huey.enqueue(pipeline)
    _ = [t.get(blocking=True) for t in result_group]
    _, zone_data = load_zone_data.call_local(science_config=science_config, optimisation_datetime=date_time)
    gc = GeneticControllerParallelCpu(
        zone_data=zone_data,
        configuration=science_config.genetic_config,
    )
    t1= time()
    gc.execute()
    optimisation_time = time() - t1
    best_phenotype = gc.best_phenotype

    result = OptimisationResult(
        best_phenotype=best_phenotype,
        optimization_time=optimisation_time,
        science_config=science_config,
    )
    return result

def optimise_zone_without_data_preparation(zone_id: str, date_time_string: str,  topography: str = TopographyType.ISLAND_ONE_WAY_RING):
    """Run optimisation without data preparation for given zone and date."""
    date_time: dt.datetime = dt.datetime.strptime(date_time_string, date_time_format)
    daf = DataAccessFacadeBasic()
    science_config: ScienceConfig = daf.get_science_config_by_zone_id(zone_id=zone_id)
    science_config.genetic_config.topography_type = topography
    _, zone_data = load_zone_data.call_local(science_config=science_config, optimisation_datetime=date_time)
    gc = GeneticControllerParallelCpu(
        zone_data=zone_data,
        configuration=science_config.genetic_config,
    )
    t1 = time()
    gc.execute()
    optimisation_time = time() - t1
    best_phenotype = gc.best_phenotype

    result = OptimisationResult(
        best_phenotype=best_phenotype,
        optimization_time=optimisation_time,
        science_config=science_config,
    )
    daf = DataAccessFacadeBasic()

    m = create_map(best_phenotype, zone_data, science_config)
    daf.save_cpu_optimisation_result(result, date_time, m)
    return result

if __name__ == "__main__":
    # Example usage
    zone_id = "lodz_synthetic"
    example_topography = TopographyType.ISLAND_ONE_WAY_RING
    date_time_string = "2021-02-28 00:00:00"
    r = optimise_zone_without_data_preparation(zone_id, date_time_string, example_topography.value)
    print(r)

