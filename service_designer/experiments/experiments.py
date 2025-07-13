import logging
from copy import deepcopy
from typing import List

import logging
from typing import List

import numpy as np

from service_designer.data.data_config import ExperimentConfig
from service_designer.data.generate_subzones import SubzonesGenerator
from service_designer.data.load_data import load_base_data
from service_designer.experiments.kpis import PriceList
from service_designer.simulator.parallel_simulator import (
    ParallelSimulateIn,
)
from service_designer.simulator.simulator_daybyday import (
    SimulateIn,
    Vehicle,
)
from service_designer.tasks.simulate import simulate
from service_designer.tools import cells_order_by_demand

log = logging.getLogger("service_designer")


class Experiment:
    """
    Class of Experiment storing experiments configuration and execute simulations.
    """

    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config: ExperimentConfig = experiment_config

    def run_experiment(self):
        """This function runs experiment"""
        log.info("loading base data")
        base_data = load_base_data(self.experiment_config)
        subzones_list = SubzonesGenerator(self.experiment_config).generate_subzones()
        price_list = PriceList(self.experiment_config.price_list_config.model_dump())

        simulate_in_list: List[ParallelSimulateIn] = []
        for subzone in subzones_list:
            subzone.generate_zone_mask(base_data.cell_ids)
            for vehicles_nbr in self.experiment_config.vehicle_count_set:
                log.info(
                    f"putting vehicles to best cells for vehicles number {vehicles_nbr} and subzone {subzone.id}"
                )
                vehicles = self._put_vehicles_to_best_cells(
                    base_data.rent_demand, subzone.zone_mask, vehicles_nbr
                )
                for (
                    service_actions_nbr
                ) in self.experiment_config.service_action_count_set:
                    if self.experiment_config.relocation_actions_proportion_set:
                        for (
                            relocation_actions_proportion
                        ) in self.experiment_config.relocation_actions_proportion_set:
                            relocations_nbr = int(
                                service_actions_nbr * relocation_actions_proportion
                            )
                            simulate_in = SimulateIn(
                                self.experiment_config.name,
                                self.experiment_config.simulator_config.model_dump(),
                                deepcopy(base_data),
                                subzone,
                                vehicles,
                                service_actions_nbr,
                                relocations_nbr,
                            )
                            parallel_simulate_in = ParallelSimulateIn(
                                simulate_in, price_list
                            )
                            simulate_in_list.append(parallel_simulate_in)
                            log.info(
                                f"creating simulation for vehicles number {vehicles_nbr}, service actions number {service_actions_nbr}, relocation actions proportion {relocation_actions_proportion}"
                            )
                    else:
                        simulate_in = SimulateIn(
                            self.experiment_config.name,
                            self.experiment_config.simulator_config.model_dump(),
                            deepcopy(base_data),
                            subzone,
                            vehicles,
                            service_actions_nbr,
                            None,
                        )
                        parallel_simulate_in = ParallelSimulateIn(simulate_in, price_list)
                        simulate_in_list.append(parallel_simulate_in)
                        log.info(
                            f"creating simulation for vehicles number {vehicles_nbr}, service actions number {service_actions_nbr}"
                        )
        simulations_count = len(simulate_in_list)
        if simulations_count == 0:
            log.info("No simulations to run. Check Experiment configuration.")
        else:
            for si in simulate_in_list:
                simulate(si)

    def _put_vehicles_to_best_cells(
        self, rent_demand: np.array, zone_mask: np.array, vehicles_nbr: int
    ) -> List[Vehicle]:
        """This function builds list of vehicles located in best cells"""

        cells_order = cells_order_by_demand(rent_demand, zone_mask)
        zone_size = len(cells_order)
        init_cells_nbr = int(vehicles_nbr / 10)
        if zone_size > 0:
            cells = [
                cells_order[i % zone_size % init_cells_nbr]
                for i in range(0, vehicles_nbr)
            ]
        else:
            cells = [0 for _ in range(0, vehicles_nbr)]
        assert len(np.unique(cells)) <= zone_size
        fuels = [
            self.experiment_config.simulator_config.model_dump()["tank_capacity"]
            for _ in range(0, vehicles_nbr)
        ]
        vehicles = self._build_vehicles_list(cells, fuels)
        return vehicles

    def _build_vehicles_list(
        self, cells: List[int], fuels: List[float]
    ) -> List[Vehicle]:
        """This function builds list of vehicles"""
        assert len(cells) == len(fuels)
        vehicles = []
        for vehicle_idx in range(len(cells)):
            cell_idx = cells[vehicle_idx]
            fuel = fuels[vehicle_idx]
            v = Vehicle(vehicle_idx, cell_idx, fuel)
            vehicles.append(v)
        return vehicles

