import logging
from typing import Dict, List

from fleet_manager.data_model.genetic_model import Action, Phenotype
from fleet_manager.data_model.zone_model import ZoneData
from fleet_manager.genetic_output_builder.genetic_output_builder import GeneticOutputBuilder

log = logging.getLogger("fleet_manager")


def build_service_team_tasks(
    best_phenotype: Phenotype, zone_data: ZoneData, optimization_result_id: int
) -> List[CreateTaskRequest]:
    """Build list of CreateTaskRequest from the best phenotype and zone_data.

    :param best_phenotype: Phenotype
    :type best_phenotype: Phenotype

    :param zone_data: zone_data
    :type zone_data: ZoneData

    :param optimization_result_id: id of optimization result
    :type optimization_result_id: int

    :return: list of create task requests
    :rtype: List[CreateTaskRequest]
    """
    output_builder = GeneticOutputBuilder()
    log.info(f"initializing service team task output builder")
    output_builder.initialize(zone_data, best_phenotype)
    vehicles_actions: List[Action] = output_builder.vehicles_actions
    requests: List[] = []
    for action in vehicles_actions:
        log.info(f"creating request for action {action}")
        if _is_relocation(action):
            requests.append(_build_relocation_request(action, optimization_result_id, zone_data.zone_id))
        elif _is_refuel(action):
            requests.append(_build_refuel_request(action, optimization_result_id, zone_data.zone_id))
        else:
            raise NotImplementedError("Action not supported")

    return requests


def _is_relocation(action: Action) -> bool:
    """
    Return True if action is relocation.

    :param action: action

    :return: True if action is relocation
    """

    return not _is_refuel(action)


def _is_refuel(action: Action) -> bool:
    """
    Return True if action is refuel.

    :param action: action

    :return: True if action is refuel
    """
    return action.station_details is not None


def _build_relocation_request(action: Action, optimization_result_id: int, zone_id: int) -> CreateTaskRequest:
    """
    Build `CreateTaskRequest` object for relocation action.

    :param action: action
    :param optimization_result_id: optimization result id

    :return: `CreateTaskRequest` object for relocation action
    """

    title: str = f"Team {action.service_team_id} Action {action.service_order} Relocate {action.vehicle_foreign_id}"

    content: Dict = {
        "zone_id": zone_id,
        "vehicle_foreign_id": action.vehicle_foreign_id,
        "action_type": "relocation",
        "source_lat": action.source_details["source_lat"],
        "source_lon": action.source_details["source_lon"],
        "source_address": action.source_details["source_address"],
        "destination_lat": action.destination_details["destination_lat"],
        "destination_lon": action.destination_details["destination_lon"],
    }

    return CreateTaskRequest(
        title=title,
        vehicle=action.vehicle_id,
        team=action.service_team_id,
        result=optimization_result_id,
        content=content,
    )


def _build_refuel_request(action: Action, optimization_result_id: int, zone_id: int) -> CreateTaskRequest:
    """
    Build `CreateTaskRequest` object for refuel action.

    :param action: action
    :param optimization_result_id: optimization result id

    :return: `CreateTaskRequest` object for refuel action
    """

    title: str = f"Team {action.service_team_id} Action {action.service_order} Refuel {action.vehicle_foreign_id}"

    content: Dict = {
        "zone_id": zone_id,
        "vehicle_foreign_id": action.vehicle_foreign_id,
        "action_type": "refuel",
        "source_lat": action.source_details["source_lat"],
        "source_lon": action.source_details["source_lon"],
        "source_address": action.source_details["source_address"],
        "station_name": action.station_details["station_name"],
        "station_lat": action.station_details["station_lat"],
        "station_lon": action.station_details["station_lon"],
        "station_address": action.station_details["station_address"],
        "destination_lat": action.destination_details["destination_lat"],
        "destination_lon": action.destination_details["destination_lon"],
    }

    return CreateTaskRequest(
        title=title,
        vehicle=action.vehicle_id,
        team=action.service_team_id,
        result=optimization_result_id,
        content=content,
    )
