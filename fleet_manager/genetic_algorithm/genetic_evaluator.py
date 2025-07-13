import copy
import logging
from operator import itemgetter
from typing import List

from fleet_manager.data_model.genetic_model import ActionType, Chromosome, Gene, Phenotype
from fleet_manager.data_model.zone_model import Cell, PowerStation, ServiceTeam, Vehicle, ZoneData

log = logging.getLogger(__name__)


class Evaluator:
    """Class responsible for evaluating specimen in a genetinc algorithm."""

    def __init__(self, zone_data: ZoneData):
        self.zone_data: ZoneData = zone_data
        self.involved_cells: List[Cell] = []
        self.moved_vehicles: List[Vehicle] = []
        self.power_stations: List[PowerStation] = zone_data.power_station
        self.base_revenue: float = zone_data.base_revenue
        self.cells = zone_data.cells
        self.revenue: float = 0
        self.cost: float = 0
        self.penalty: float = 0
        self.refuel_bonus: float = 0
        self.recharge_bonus: float = 0
        self._bonus_wait_time: float = 0

    def evaluate_phenotype(self, phenotype: Phenotype):
        """
        Evaluate provided phenotype in place (save fitness in Phenotype)z .
        :param phenotype: Phenotype object
        """
        self.involved_cells = []
        self.moved_vehicles = []
        for power_station in self.power_stations:
            power_station.charging_vehicles = copy.deepcopy(power_station.base_charging_vehicles)
        self.revenue = 0
        self.cost = 0
        self.penalty = 0
        self.refuel_bonus = 0
        self._bonus_wait_time = 0

        for ch_i in range(phenotype.genome.number_chromosome_1):
            self._develop_chromosome_1(phenotype.genome[ch_i])
        for ch_i in range(phenotype.genome.number_chromosome_2):
            self._develop_chromosome_2(phenotype.genome[phenotype.genome.number_chromosome_1 + ch_i])

        self._compute_charge_station_occupation()
        self._compute_bonuses()
        self._compute_revenue()
        phenotype.base_revenue = self.base_revenue
        phenotype.diff_revenue = self.revenue
        phenotype.cost = -self.cost
        phenotype.penalty = -self.penalty
        phenotype.bonus = self.refuel_bonus
        phenotype.fitness = self.revenue - self.cost - self.penalty + self.refuel_bonus + self.recharge_bonus

    def _develop_chromosome_1(self, chromosome: Chromosome):
        """
        Calculate metrics for a 1p team Chromosome.
        :param chromosome: Chromosome object
        """
        chromosome.total_time = 0
        chromosome.approach_distance = 0
        chromosome.relocation_distance = 0

        if chromosome.get_length() == 0:
            return

        service_team: ServiceTeam = self.zone_data.service_team[chromosome.service_team_index]
        next_cell_id: int = chromosome[0].source_cell_id
        chromosome.approach_distance += self.zone_data.distance_cell_cell[service_team.start_cell_id, next_cell_id]
        chromosome.total_time += self.zone_data.time_cell_cell[service_team.start_cell_id, next_cell_id]

        for g_i in range(chromosome.get_length()):
            next_cell_id = (
                service_team.end_cell_id if g_i + 1 == chromosome.get_length() else chromosome[g_i + 1].source_cell_id
            )
            self._develop_gene_1(chromosome, chromosome[g_i], next_cell_id)

        self._compute_costs(chromosome)
        self._compute_penalties(chromosome)

    def _develop_chromosome_2(self, chromosome: Chromosome):
        """
        Calculate metrics for a 2p team Chromosome.
        :param chromosome: Chromosome object
        """
        chromosome.total_time = 0
        chromosome.approach_distance = 0
        chromosome.relocation_distance = 0

        service_team: ServiceTeam = self.zone_data.service_team[chromosome.service_team_index]
        previous_cell_id: int = service_team.start_cell_id

        for g_i in range(chromosome.get_length()):
            self._develop_gene_2(chromosome, chromosome[g_i], previous_cell_id)
            previous_cell_id = chromosome[g_i].destination_cell_id

        # add time and distance from the last vehicle to the end depot
        end_depot_cell_id: int = service_team.end_cell_id
        chromosome.approach_distance += self.zone_data.distance_cell_cell[previous_cell_id, end_depot_cell_id]
        chromosome.total_time += self.zone_data.time_cell_cell[previous_cell_id, end_depot_cell_id]

        self._compute_costs(chromosome)
        self._compute_penalties(chromosome)

    def _develop_gene_1(self, chromosome: Chromosome, gene: Gene, next_cell_id: int):
        """
        Process action for 1p team.
        :param chromosome: Chromosome object (for 1p team)
        :param gene: Gene object
        :param next_cell_id: id of the next cell (int)
        """
        vehicle: Vehicle = self.zone_data.fleet[gene.vehicle_id]
        source_cell: Cell = self.cells[gene.source_cell_id]
        destination_cell: Cell = self.cells[next_cell_id]

        self._process_action_refueling(chromosome, vehicle, source_cell, destination_cell)
        self._involve_cell(source_cell, destination_cell, gene, vehicle)

    def _develop_gene_2(self, chromosome: Chromosome, gene: Gene, previous_cell_id: int):
        """
        Process action for 2p team.
        :param chromosome: Chromosome object (for 2p team)
        :param gene: Gene object
        :param previous_cell_id: id of the previous cell (int)
        """
        vehicle: Vehicle = self.zone_data.fleet[gene.vehicle_id]
        source_cell: Cell = self.cells[gene.source_cell_id]
        destination_cell: Cell = self.cells[gene.destination_cell_id]

        # travel from previous location to current vehicle
        chromosome.approach_distance += self.zone_data.distance_cell_cell[previous_cell_id, gene.source_cell_id]
        chromosome.total_time += self.zone_data.time_cell_cell[previous_cell_id, gene.source_cell_id]

        if gene.action_type == ActionType.F:
            self._process_action_refueling(chromosome, vehicle, source_cell, destination_cell)
        elif gene.action_type == ActionType.P:
            self._process_action_plugging(chromosome, vehicle, source_cell, destination_cell, gene.power_station_id)
        elif gene.action_type == ActionType.U:
            chromosome.relocation_distance += self.zone_data.distance_cell_cell[
                gene.source_cell_id, gene.destination_cell_id
            ]
            chromosome.total_time += self.zone_data.end_charge_time
            # vehicle.end_charging_time = chromosome.total_time
            charging_vehicle: dict = dict(
                vehicle_id=gene.vehicle_id, action_time=chromosome.total_time, action_type=ActionType.U
            )
            self.power_stations[vehicle.power_station_id].charging_vehicles.append(charging_vehicle)
            chromosome.total_time += self.zone_data.time_cell_cell[gene.source_cell_id, gene.destination_cell_id]
            self._compute_range_factor_for_electric(vehicle)
            chromosome.unplugging_number += 1
        elif gene.action_type == ActionType.C:
            chromosome.relocation_distance += self.zone_data.distance_cell_cell[
                gene.source_cell_id, gene.destination_cell_id
            ]
            chromosome.total_time += self.zone_data.end_charge_time
            # vehicle.end_charging_time = chromosome.total_time
            charging_vehicle: dict = dict(
                vehicle_id=gene.vehicle_id, action_time=chromosome.total_time, action_type=ActionType.U
            )
            self.power_stations[vehicle.power_station_id].charging_vehicles.append(charging_vehicle)
            chromosome.total_time += self.zone_data.time_cell_cell[gene.source_cell_id, gene.destination_cell_id]
            self._compute_range_factor_for_electric(vehicle)
        elif gene.action_type == ActionType.R:
            self._process_action_relocation(chromosome, vehicle, source_cell, destination_cell)
        else:
            raise ValueError("Unknown action type!")

        self._involve_cell(source_cell, destination_cell, gene, vehicle)

    def _process_action_refueling(
        self, chromosome: Chromosome, vehicle: Vehicle, source_cell: Cell, destination_cell: Cell
    ):
        """
        Process action refueling.

        :param chromosome: chromosome
        :param vehicle: vehicle
        :param source_cell: source cell
        :param destination_cell: destination cell
        """

        chromosome.relocation_distance += self.zone_data.distance_cell_station_cell[
            source_cell.cell_id, destination_cell.cell_id
        ]
        chromosome.total_time += (
            chromosome.refuel_time
            + self.zone_data.time_cell_station_cell[source_cell.cell_id, destination_cell.cell_id]
        )

        vehicle.revenue_range_factor = 1
        self.refuel_bonus += 1 - vehicle.base_revenue_range_factor
        chromosome.refuelings_number += 1

    def _process_action_relocation(
        self, chromosome: Chromosome, vehicle: Vehicle, source_cell: Cell, destination_cell: Cell
    ):
        """
        Process action relocation.

        :param chromosome: chromosome
        :param vehicle: vehicle
        :param source_cell: source cell
        :param destination_cell: destination cell
        """
        chromosome.relocation_distance += self.zone_data.distance_cell_cell[
            source_cell.cell_id, destination_cell.cell_id
        ]
        chromosome.total_time += (
            chromosome.relocation_time + self.zone_data.time_cell_cell[source_cell.cell_id, destination_cell.cell_id]
        )
        self._bonus_wait_time += 1 - vehicle.base_revenue_time_factor
        chromosome.relocations_number += 1

    def _process_action_plugging(
        self, chromosome: Chromosome, vehicle: Vehicle, source_cell: Cell, destination_cell: Cell, power_station_id: int
    ):
        """
        Process action plugging. Add vehicle to the list of vehicles waiting for charging on the nearest charging
        station.

        :param chromosome: chromosome
        :param vehicle: vehicle
        :param source_cell: source cell
        :param destination_cell: destination cell
        :param power_station_id: id of the power station
        """

        vehicle.power_station_id = power_station_id
        chromosome.relocation_distance += self.zone_data.distance_cell_cell[
            source_cell.cell_id, destination_cell.cell_id
        ]
        chromosome.total_time += (
            self.zone_data.begin_charge_time
            + self.zone_data.time_cell_cell[source_cell.cell_id, destination_cell.cell_id]
        )
        charging_vehicle: dict = dict(
            vehicle_id=vehicle.vehicle_index, action_time=chromosome.total_time, action_type=ActionType.P
        )
        self.power_stations[vehicle.power_station_id].charging_vehicles.append(charging_vehicle)
        vehicle.revenue_range_factor = 1
        self.recharge_bonus += 1 - vehicle.base_revenue_range_factor
        chromosome.plugging_number += 1

    def _involve_cell(self, source_cell: Cell, destination_cell: Cell, gene: Gene, vehicle: Vehicle):
        """
        Mark cells as involved and vehicle as moved.

        :param source_cell: source cell
        :param destination_cell: destination cell
        :param gene: this parameter is not used
        :param vehicle: vehicle
        """

        if not source_cell.involved:
            source_cell.vehicles = source_cell.base_vehicles.copy()
            source_cell.vehicle_number = source_cell.base_vehicle_number
            source_cell.involved = True
            self.involved_cells.append(source_cell)

        if not destination_cell.involved:
            destination_cell.vehicles = destination_cell.base_vehicles.copy()
            destination_cell.vehicle_number = destination_cell.base_vehicle_number
            destination_cell.involved = True
            self.involved_cells.append(destination_cell)

        vehicle.moved = True
        self.moved_vehicles.append(vehicle)
        vehicle.revenue_time_factor = 1

        if source_cell != destination_cell:
            source_cell.vehicles.remove(vehicle)
            destination_cell.vehicles.append(vehicle)

    def _compute_costs(self, chromosome: Chromosome):
        """
        Add service team cost to solution cost sum.

        :param chromosome: chromosome of service team
        """

        if self.zone_data.price_per_action:
            team: ServiceTeam = self.zone_data.service_team[chromosome.service_team_index]
            team_size: int = 1 if chromosome.kind == 1 else 2
            # distance cost
            self.cost += team.distance_cost * (chromosome.approach_distance + chromosome.relocation_distance) / 1000
            # time cost
            self.cost += team.price_per_refuelling * chromosome.refuelings_number

            if team_size > 1:
                self.cost += team.price_per_relocation * chromosome.relocations_number

        else:
            team: ServiceTeam = self.zone_data.service_team[chromosome.service_team_index]
            team_size: int = 1 if chromosome.kind == 1 else 2
            self.cost += (
                team.distance_cost * (chromosome.approach_distance + team_size * chromosome.relocation_distance) / 1000
                + team_size * team.time_cost * chromosome.total_time
            )

    def _compute_charge_station_occupation(self):
        """
        Charge vehicle on charge stations. Charge station has a capacity limit. If there are more vehicles on the charge
        station than the capacity of the charge station, the next vehicle will start charging after unplugging of the
        previous vehicle.
        """

        charged_vehicles: List[Vehicle] = []
        for power_station in self.power_stations:
            waiting_vehicles = []
            occupation = len(power_station.base_charging_vehicles)
            sorted_vehicles: List[dict] = sorted(power_station.charging_vehicles, key=itemgetter("action_time"))
            for s_v in sorted_vehicles:
                vehicle: Vehicle = self.zone_data.fleet[s_v["vehicle_id"]]
                if vehicle not in charged_vehicles:
                    charged_vehicles.append(vehicle)
                if s_v["action_type"] == ActionType.P:
                    if occupation < power_station.capacity:
                        vehicle.start_charging_time = s_v["action_time"]
                        occupation += 1
                    else:
                        waiting_vehicles.append(vehicle)
                if s_v["action_type"] == ActionType.U:
                    vehicle.end_charging_time = s_v["action_time"]
                    if vehicle.start_charging_time != -1:
                        if len(waiting_vehicles) > 0:
                            vehicle = waiting_vehicles.pop(0)
                            vehicle.start_charging_time = s_v["action_time"]
                        else:
                            occupation -= 1

        for vehicle in charged_vehicles:
            self._compute_range_factor_for_electric(vehicle)

    def _compute_revenue(self):
        """
        Compute solution revenue as a sum of cells revenue. Revenue of each cell is a function of vehicles number in
        that cell after optimisation.
        """

        for cell in self.involved_cells:
            cell_revenue: float = 0
            factors: float = 0
            vehicle_number: int = 0
            for vehicle in cell.vehicles:
                if vehicle.revenue_range_factor > 0 and vehicle.revenue_time_factor > 0:
                    factors += vehicle.revenue_range_factor * vehicle.revenue_time_factor
                    vehicle_number += 1
            if vehicle_number > 0:
                if vehicle_number > 10:
                    vehicle_number = 10
                cell_revenue: float = factors * (
                    self.zone_data.revenue[cell.cell_id, vehicle_number - 1] / vehicle_number
                )
                if vehicle_number == 1:
                    cell_revenue *= 0.95
                elif vehicle_number == 2:
                    cell_revenue *= 0.9

            self.revenue += cell_revenue - cell.base_revenue
            cell.involved = False  # cleaning

        for vehicle in self.moved_vehicles:
            vehicle.revenue_range_factor = vehicle.base_revenue_range_factor
            vehicle.revenue_time_factor = vehicle.base_revenue_time_factor
            vehicle.start_charging_time = -1
            vehicle.end_charging_time = -1
            vehicle.moved = False  # cleaning

    def _compute_penalties(self, chromosome: Chromosome):
        """
        Compute penalty due to exceeded work time of the service teams.
        """

        planned_time_work_sec: int = self.zone_data.service_team[chromosome.service_team_index].planned_time_work * 60
        if planned_time_work_sec < chromosome.total_time:
            self.penalty += (
                chromosome.total_time - planned_time_work_sec
            ) * self.zone_data.service_exceeded_time_penalty

    def _compute_bonuses(self):
        """
        Sum configuration bonuses.
        """

        self.refuel_bonus *= self.zone_data.service_refuel_bonus
        self._bonus_wait_time *= self.zone_data.service_wait_time_bonus
        self.refuel_bonus += self._bonus_wait_time

    def _compute_range_factor_for_electric(self, vehicle: Vehicle) -> None:
        """Computes electric vehicle revenue reduction based on its range."""
        if vehicle.start_charging_time > -1:
            if vehicle.end_charging_time > -1:
                charging_time: int = vehicle.end_charging_time - vehicle.start_charging_time
                if charging_time < 0:
                    charging_time = 0
                vehicle.revenue_range_factor = self.zone_data.compute_range_factor(
                    int(vehicle.range + vehicle.charging_factor * charging_time)
                )
            else:
                vehicle.revenue_range_factor = 1
