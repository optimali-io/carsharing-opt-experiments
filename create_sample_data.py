import json

import fiona

from config import settings
from fleet_manager.data_model.config_models import ScienceConfig, GeneticConfiguration, Zone, ZoneDataConfig
from fleet_manager.data_model.zone_model import ServiceTeam, VehicleModels, VehicleModelName, VehicleModel, FuelType


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
            "population_size": 100,
            "maximum_time": 1200,
            "maximum_no_improvement": 500,
            "maximum_generation_number": 5000,
            "maximum_tune_generation_number": 200,
            "island_population": 25,
            "n_neighbors": 10,
            "tolerance": 0.01,
            "migration_interval": 100,
            "migration_rate": 0.1,
            "topography_type": "IslandTwoWayRing",
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
        zone_data_config=ZoneDataConfig()
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
            service_team_id=1,
            service_team_name="1-person team",
            service_team_kind=1,
            planned_time_work=600,
            time_cost=0.008333333,
            distance_cost=0.4,
            start_cell_id=60,
            end_cell_id=140,
        ).model_dump(),
        ServiceTeam(
            service_team_id=2,
            service_team_name="2-person team",
            service_team_kind=2,
            planned_time_work=600,
            time_cost=0.008333333,
            distance_cost=0.4,
            start_cell_id=60,
            end_cell_id=140,
        ).model_dump()
    ]

    with open(f"{settings.ZONES_PATH}/{zone_id}/service_teams.json", "w") as f:
        json.dump(service_teams, f, indent=4)

def create_default_vehicle_models(zone_id: str) -> None:
    """
    Function creates default vehicle models for the given zone.
    :param zone_id: str, id of the zone
    """
    models =VehicleModels(
        models={
            VehicleModelName.yaris: VehicleModel(
                average_fuel_consumption=0.001,
                tank_capacity=50,
                fuel_type=FuelType.PETROL
            )
        }
    )
    m = models.model_dump()

    with open(f"{settings.ZONES_PATH}/{zone_id}/vehicle_models.json", "w") as f:
        json.dump(m, f)


create_default_science_config("lodz")