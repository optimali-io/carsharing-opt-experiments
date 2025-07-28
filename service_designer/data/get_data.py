import logging
from datetime import datetime

from config import huey
from core.tasks.data_preparation import (
    calculate_historical_demand_for_date_range,
    calculate_directions,
    create_nearest_petrol_station_distance,
    calculate_demand_by_weekday,
)
from core.tasks.distance_time_modification import (
    run_distance_modification_approximation,
    run_traffic_factors_task,
)
from service_designer.data.data_config import ExperimentConfig

logger = logging.getLogger("service_designer")


def prepare_base_data_if_not_exist(experiment_config: ExperimentConfig):
    """Run data preparation task for each missing experiment component and wait for them to finish."""

    base_zone_id = experiment_config.data_config.base_zone.name
    data_config_zone_id = experiment_config.get_experiment_zone_id()

    logger.info(
        f"Generating base data for DataConfig {experiment_config.data_config.name}"
    )

    start_date = experiment_config.data_config.start_date
    date = experiment_config.data_config.end_date

    long_date_start = experiment_config.data_config.approximation_directions_start_date
    long_date_end = (
        experiment_config.data_config.approximation_directions_end_date or date
    )

    blur_factor = experiment_config.data_config.blur_factor
    blur_distance = experiment_config.data_config.blur_distance

    cell_ids = experiment_config.data_config.base_zone.cell_ids

    pipeline = (
        calculate_historical_demand_for_date_range.s(
            zone_id=base_zone_id,
            cell_ids=cell_ids,
            start_date=long_date_start,
            end_date=long_date_end,
            output_zone_id=data_config_zone_id,
        )
        .then(
            calculate_demand_by_weekday,
            zone_id=base_zone_id,
            today=date,
            blur_factor=blur_factor,
            blur_distance=blur_distance,
            output_zone_id=data_config_zone_id,
            from_date=long_date_start,
            cell_ids=cell_ids,
        )
        .then(
            calculate_directions,
            zone_id=base_zone_id,
            today=long_date_end,
            start_date=long_date_start,
            output_zone_id=data_config_zone_id,
        )
        .then(
            run_distance_modification_approximation,
            zone_id=base_zone_id,
            cell_ids=cell_ids,
            today=long_date_end,
            start_date=long_date_start,
            output_zone_id=data_config_zone_id,
        )
        .then(
            run_traffic_factors_task,
            zone_id=base_zone_id,
            today=long_date_end,
            start_date=long_date_start,
            output_zone_id=data_config_zone_id,
        )
        .then(
            create_nearest_petrol_station_distance(
                zone_id=base_zone_id,
                cell_ids=cell_ids,
                start_date=datetime(start_date.year, start_date.month, start_date.day, hour=23, minute=59),
                out_zone=data_config_zone_id,
            )
        )

    )

    result = huey.enqueue(pipeline)
    result.get(blocking=True)
