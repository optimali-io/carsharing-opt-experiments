import json

import fiona

from config import settings
from fleet_manager.data_model.config_models import (
    ScienceConfig,
    GeneticConfiguration,
    Zone,
    ZoneDataConfig,
)
from fleet_manager.data_model.zone_model import (
    ServiceTeam,
    VehicleModels,
    VehicleModelName,
    VehicleModel,
    FuelType,
)


def get_zone_cell_ids(zone_id: str) -> list[str]:
    """
    Function returns list of cell ids for given zone.
    :param zone_id: str, id of the zone
    :return: list[int], list of cell ids
    """
    grid_path = f"{settings.ZONES_PATH}/{zone_id}/zone_grid.shp"
    with fiona.open(grid_path, "r") as shp:
        return [feat["properties"]["id"] for feat in shp]


def create_default_science_config(zone_id: str) -> ScienceConfig:
    """
    Function creates default ScienceConfig object.
    :return: ScienceConfig, default ScienceConfig object
    """
    genetic_config = GeneticConfiguration.model_validate(
        {
            "population_size": 25,
            "maximum_time": 1200,
            "maximum_no_improvement": 500,
            "maximum_generation_number": 5000,
            "maximum_tune_generation_number": 200,
            "island_population": 25,
            "n_neighbors": 10,
            "tolerance": 0.01,
            "migration_interval": 100,
            "migration_rate": 0.1,
            "topography_type": "IslandOneWayRing",
            "chromosome_deviation": 15,
            "tournament_samples": 4,
            "probability_gene_mutation": 0.03,
            "probability_action_change": 0.1,
            "mutation_types": ["move", "swap", "add", "del", "change"],
            "weight_mutation_types": [1, 10, 5, 5, 4],
            "tuned_mutation_types": ["move", "swap", "add", "del", "change"],
            "tuned_weight_mutation_types": [1, 10, 5, 5, 4],
        }
    )

    cell_ids = get_zone_cell_ids(zone_id)
    sc = ScienceConfig(
        genetic_config=genetic_config,
        zone=Zone(id=zone_id, cell_ids=cell_ids),
        zone_data_config=ZoneDataConfig(service_exceeded_time_penalty=0.2),
    )

    with open(f"{settings.ZONES_PATH}/{zone_id}/science_config.json", "w") as f:
        f.write(sc.model_dump_json(indent=4))

    return sc


def create_default_service_teams(zone_id: str) -> None:
    """
    Function creates default service teams for the given zone.
    :param zone_id: str, id of the zone
    """
    service_teams = [
        ServiceTeam(
            service_team_id=0,
            service_team_name="1-person team",
            service_team_kind=1,
            planned_time_work_minutes=600,
            time_cost_per_second=0.016,
            distance_cost_per_km=0.4,
            start_cell_id=60,
            end_cell_id=140,
        ).model_dump(),
        ServiceTeam(
            service_team_id=1,
            service_team_name="2-person team 1",
            service_team_kind=2,
            planned_time_work_minutes=600,
            time_cost_per_second=0.016,
            distance_cost_per_km=0.4,
            start_cell_id=60,
            end_cell_id=140,
        ).model_dump(),
        ServiceTeam(
            service_team_id=2,
            service_team_name="2-person team 2",
            service_team_kind=2,
            planned_time_work_minutes=600,
            time_cost_per_second=0.016,
            distance_cost_per_km=0.4,
            start_cell_id=60,
            end_cell_id=140,
        ).model_dump(),
    ]

    with open(f"{settings.ZONES_PATH}/{zone_id}/service_teams.json", "w") as f:
        json.dump(service_teams, f, indent=4)


def create_default_vehicle_models(zone_id: str) -> None:
    """
    Function creates default vehicle models for the given zone.
    :param zone_id: str, id of the zone
    """
    models = VehicleModels(
        models={
            VehicleModelName.yaris: VehicleModel(
                average_fuel_consumption=8, tank_capacity=50, fuel_type=FuelType.PETROL
            )
        }
    )
    m = models.model_dump()

    with open(f"{settings.ZONES_PATH}/{zone_id}/vehicle_models.json", "w") as f:
        json.dump(m, f)


def create_default_sdesigner_experiment_config(zone_id: str) -> None:
    """
    Function creates default service designer experiment config for the given zone.
    :param zone_id: str, id of the zone
    """
    from service_designer.data.data_config import ExperimentConfig
    experiment_id = "lodz_synth_experyment"
    cell_ids = get_zone_cell_ids(zone_id)
    raw_config = {
        "name": experiment_id,
        "type": "evaluation",
        "data_config": {
            "name": "lodz_dataconfig",
            "base_zone": {
                "name": zone_id,
                "timezone": "Europe/Warsaw",
                "cell_ids": cell_ids,
            },
            "start_date": "2021-02-01",
            "end_date": "2021-02-28",
            "approximation_directions_start_date": "2021-02-01",
            "approximation_directions_end_date": "2021-02-28",
            "include_holidays": True,
            "rent_max_distance": 100,
            "blur_factor": 0.07,
            "blur_distance": 800.0,
        },
        "generate_subzones": False,
        "subzones_generator_config": {
            "name": "lodz_subzones_generator",
            "subzone_cells_origin": "by_demand",
            "subzone_count": 10,
            "minimum_cells_in_subzone": 100,
        },
        "subzones_set": None,
        "vehicle_count_set": [100],
        "service_action_count_set": [15],
        "relocation_actions_proportion_set": None,
        "simulator_config": {
            "weeks_of_simulations": 10,
            "exclude_first_weeks": 0,
            "tank_capacity": 36.0,
            "fuel_usage": 6.0,
            "refueling_start_level": 6.0,
            "stop_renting_fuel_level": 6.0,
        },
        "price_list_config": {
            "price_per_rent_start": 0.0,
            "price_per_km": 0.8,
            "price_per_minute": 0.5,
            "cost_per_action_km": 0.3,
            "service_factor": 2.0,
            "cost_per_relocation": 15.0,
            "cost_per_refueling": 35.0,
        },
        "simulations_mode": True,
        "real_statistics_mode": False,
    }
    experiment_config = ExperimentConfig.model_validate(raw_config)
    pth = f"{settings.SDESIGNER_DIR}/experiment_configs/experiment_config_{experiment_id}.json"
    with open(pth, "w") as f:
        f.write(experiment_config.model_dump_json(indent=4))


if __name__ == "__main__":
    zone_id = "lodz_synthetic"
    if not zone_id:
        raise ValueError("Please provide a valid zone_id.")
    create_default_science_config(zone_id)
    create_default_service_teams(zone_id)
    create_default_vehicle_models(zone_id)
    create_default_sdesigner_experiment_config(zone_id)
