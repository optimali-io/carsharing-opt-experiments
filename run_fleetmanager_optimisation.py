import datetime as dt
import logging
from datetime import timedelta

from config import huey
from core.db.data_access_facade_basic import DataAccessFacadeBasic
from core.tasks.data_preparation import calculate_demand_prediction, calculate_historical_demand_for_date_range, \
    calculate_directions, calculate_revenue_matrix
from core.tasks.distance_time_modification import run_distance_modification_approximation, run_traffic_factors_task
from fleet_manager.data_model.config_models import ScienceConfig
from fleet_manager.tasks.optimisation import load_zone_data, run_genetic_algorithm

date_format = "%Y-%m-%d"
date_time_format = "%Y-%m-%d %H:%M:%S"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
)


def optimise_zone_with_data_preparation(zone_id: str, date_time_string: str):
    """Run optimisation with data preparation for given zone and date."""
    date_time: dt.datetime = dt.datetime.strptime(date_time_string, date_time_format)
    date: dt.date = date_time.date()
    daf = DataAccessFacadeBasic()
    science_config: ScienceConfig = daf.get_science_config_by_zone_id(zone_id=zone_id)
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
        .then(load_zone_data, science_config=science_config, optimisation_datetime=date_time)
        .then(run_genetic_algorithm, optimisation_datetime=date_time)
    )
    huey.enqueue(pipeline)

def optimise_zone_without_data_preparation(zone_id: str, date_time_string: str):
    """Run optimisation without data preparation for given zone and date."""
    date_time: dt.datetime = dt.datetime.strptime(date_time_string, date_time_format)
    daf = DataAccessFacadeBasic()
    science_config: ScienceConfig = daf.get_science_config_by_zone_id(zone_id=zone_id)
    pipeline = (
        load_zone_data.s(science_config=science_config, optimisation_datetime=date_time)
        .then(run_genetic_algorithm, optimisation_datetime=date_time)
    )
    huey.enqueue(pipeline)

if __name__ == "__main__":
    # Example usage
    zone_id = "lodz_synthetic"
    date_time_string = "2021-02-28 00:00:00"
    optimise_zone_without_data_preparation(zone_id, date_time_string)
    print(f"Optimisation for zone {zone_id} on {date_time_string} started. Switch to the Huey consumer to monitor progress.")

