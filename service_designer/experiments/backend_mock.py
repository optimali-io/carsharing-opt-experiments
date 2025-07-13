from typing import Any, Dict

from service_designer.external.backend import Backend


class BackendMock(Backend):
    """
    Implements `Backend` interface for tests.
    """

    def __init__(self):
        self._base_zone = {"id": 1, "cell_ids": ["C0", "C1", "C2", "C3", "C4"]}
        self._data_config = {"id": 1, "base_zone": self._base_zone}
        self._experiment = {
            "id": 1,
            "name": "Test",
            "type": "gridsearch",
            "data_config": self._data_config,
            "subzones_set": [1],
            "vehicle_count_set": [100],
            "service_action_count_set": [15],
            "relocation_actions_proportion_set": [],
            "simulations_mode": True,
            "real_statistics_mode": False,
            "weeks_of_simulations": 52,
            "tank_capacity": "36.00",
            "fuel_usage": "6.00",
            "refueling_start_level": "6.00",
            "stop_renting_fuel_level": "6.00",
            "exlude_first_weeks": 0,
            "price_per_rent_start": "0.00",
            "price_per_km": "0.80",
            "price_per_minute": "0.50",
            "cost_per_action_km": "0.30",
            "cost_per_relocation": "15.00",
            "cost_per_refueling": "35.00",
        }
        self._subzone = {
            "id": 1,
            "name": "przykład",
            "zone_generator_id": None,
            "author": 37,
            "cell_ids": ["C1", "C2", "C3"],
            "comment": "",
        }

    def get_experiment(self, id: str) -> Dict[str, Any]:
        """
        Get experiment dictionary.

        :param id: experiment id

        :returns: experiment dictionary
        """

        return self._experiment

    def get_subzone(self, id: str) -> Dict[str, Any]:
        """
        Get subzone dictionary.

        :param id: subzone id

        :returns: subzone dictionary
        """

        return self._subzone
