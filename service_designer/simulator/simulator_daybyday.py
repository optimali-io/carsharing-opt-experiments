"""
Module contains Simulator DayByDay and simulation components: Vehicle, CellQuality, Board, Rent, Relocation, Refuel.
"""

import logging
import random
from typing import Dict, List, Optional

import numpy as np


from service_designer.data.base_data import BaseData
from service_designer.data.subzones import Subzone

log = logging.getLogger("service_designer")


class Vehicle:
    """
    Vehicle objects simulating carsharing service by Simulator DayByDay.

    :param vehicle_idx: Vehicle identifier
    :param cell_idx: cell identifier set as position of Vehicle
    :param fuel: fuel level
    """

    def __init__(self, vehicle_idx: int, cell_idx: int, fuel: float):
        self.idx: int = int(vehicle_idx)
        self.cell_idx: int = int(cell_idx)
        self.fuel: float = float(fuel)
        self.history = []

    def __str__(self):
        return f"v {self.idx} c {self.cell_idx} f {self.fuel:.2f}"


class CellQuality:
    """
    Class returning quality of subzone cells determinated by demand.

    :param rent_demand: array of demand per cell
    :param act_zone_mask: subzone mask of cells
    """

    def __init__(self, rent_demand: np.array, act_zone_mask: np.array):
        self.quality_c = np.sum(rent_demand, axis=0)
        self.order_i = np.flip(np.argsort(self.quality_c), axis=0)
        self.zone_order_i = np.array(
            [i for i in self.order_i if act_zone_mask[i]], dtype=np.uint32
        )


class Board:
    """
    Class of simulator Board built from cells and stores positions of Vehicles.

    :param vehicles: list of Vehicle objects
    """

    def __init__(self, vehicles: List[Vehicle]):
        self._all_vehicles: List[Vehicle] = vehicles
        self._cell_vehicles: Dict[int, List[Vehicle]] = Board._build_cell_vehicles(
            vehicles
        )

    @staticmethod
    def _build_cell_vehicles(vehicles: List[Vehicle]) -> Dict[int, List[Vehicle]]:
        """
        Build dictionary where cell_idx is a key, and value is a List of Vehicle in the cell.

        :param vehicles: list of Vehicle objects

        :return: dictionary where cell_idx is a key, and value is a List of Vehicle in the cell
        """

        cell_vehicles = {}
        for _, v in enumerate(vehicles):
            if v.cell_idx not in cell_vehicles:
                cell_vehicles[v.cell_idx] = []
            cell_vehicles[v.cell_idx].append(v)
        return cell_vehicles

    def occupied_cells(self) -> List[int]:
        """
        Get list of indices of cells that has vehicles.

        :return: list of indices of cells that has vehicles
        """

        return self._cell_vehicles.keys()

    def vehicles_in_cell(self, cell_idx: int) -> List[Vehicle]:
        """
        Get list of the vehicles in the cell.

        :param cell_idx: cell index

        :return: list of the vehicles in the cell
        """

        return self._cell_vehicles[cell_idx]

    def vehicles_all(self) -> List[Vehicle]:
        """
        Get list of all vehicles.

        :returns: list of all vehicles
        """

        return self._all_vehicles

    def move(self, vehicle: Vehicle, dst_cell_idx: int) -> None:
        """
        Move vehicle to cell.

        :param vehicle: vehicle to move
        :param dst_cell_idx: destination cell index
        """

        src_cell_idx = vehicle.cell_idx
        vehicle.cell_idx = dst_cell_idx

        self._cell_vehicles[src_cell_idx].remove(vehicle)
        if not self._cell_vehicles[src_cell_idx]:
            del self._cell_vehicles[src_cell_idx]

        if dst_cell_idx not in self._cell_vehicles:
            self._cell_vehicles[dst_cell_idx] = []
        self._cell_vehicles[dst_cell_idx].append(vehicle)


class Rent:
    """
    Simulated Rent action.

    :param hour: hour of simulation - timestamp of Simulator
    :param vehicle_idx: id of rented Vehicle
    :param src_cell_idx: source position of Vehicle
    :param dst_cell_idx: Vehicle destionation
    :param distance: distance of rent
    :param time: time of rent
    """

    def __init__(
        self,
        hour: int,
        vehicle_idx: int,
        src_cell_idx: int,
        dst_cell_idx: int,
        distance: float,
        time: float,
    ):
        self.hour: int = hour
        self.vehicle_idx: int = vehicle_idx
        self.start_cell_idx: int = src_cell_idx
        self.stop_cell_idx: int = dst_cell_idx
        self.distance: float = distance
        self.time: float = time


class Refuel:
    """Simulated Refuel action.

    :param hour: hour of simulation - timestamp of Simulator
    :param vehicle_idx: id of rented Vehicle
    :param start_fuel: actual fuel level
    :param stop_fuel: fuel level after refueling
    :param distance_to_station: distance to station and back

    """

    def __init__(
        self,
        hour: int,
        vehicle_idx: int,
        start_fuel: float,
        stop_fuel: float,
        distance_to_station: float,
        source_cell_idx: int,
    ):
        self.hour: int = hour
        self.vehicle_idx: int = vehicle_idx
        self.start_fuel: float = start_fuel
        self.stop_fuel: float = stop_fuel
        self.distance_to_station: float = distance_to_station
        self.source_cell_idx: int = source_cell_idx


class Relocation:
    """
    Simulated Relocation action.

    :param hour: hour of simulation - timestamp of Simulator
    :param vehicle_idx: id of rented Vehicle
    :param start_cell_idx: source position of Vehicle
    :param stop_cell_idx: Vehicle destionation of relocation
    """

    def __init__(
        self,
        hour: int,
        vehicle_idx: int,
        start_cell_idx: int,
        stop_cell_idx: int,
        distance: float,
    ):
        self.hour: int = hour
        self.vehicle_idx: int = vehicle_idx
        self.start_cell_idx: int = start_cell_idx
        self.stop_cell_idx: int = stop_cell_idx
        self.distance: float = distance


class SimulateIn:
    """
    Input params of SimulatorDayByDay.

    :param config: configuration parameters collected in dictionary
    :param base_data: data matrix's of base input data of Simulator
    :param subzone: subzone with mask of cells
    :param vehicles: list of Vehicle objects
    :param service_actions_nbr: Optional number of service actions:
                                - sum of refueling actions and relocations if relocations_nbr not set
                                - number of refueling actions if relocations_nbr is set
    :param relocations_nbr: Optional number of relocations, if not set the number of relocations depends on refueling:
                            service_actions_nbr minus needed refueling actions
    """

    def __init__(
        self,
        experiment_id: str,
        config: Dict,
        base_data: BaseData,
        subzone: Subzone,
        vehicles: List[Vehicle],
        service_actions_nbr: Optional[int] = None,
        relocations_nbr: Optional[int] = None,
    ):
        self.experiment_id: str = experiment_id
        self.config = config
        self.base_data = base_data
        self.subzone = subzone

        self.vehicles = vehicles
        self.service_actions_nbr = service_actions_nbr
        self.relocations_nbr = relocations_nbr


class SimulateOut:
    """
    Output params for SimulatorDayByDay.
    """

    def __init__(self, vehicles: List[Vehicle]):
        self.vehicles: List[Vehicle] = vehicles


class Simulator:
    """
    Simulator DayByDay - simulate rents hour by hour and service actions after each day of simulation.

    :param simulate_in: input params
    """

    def __init__(self, simulate_in: SimulateIn):
        self._board: Board = Board(simulate_in.vehicles)
        self._base_data: BaseData = simulate_in.base_data
        self._cell_quality: CellQuality = CellQuality(
            simulate_in.base_data.rent_demand, simulate_in.subzone.zone_mask
        )

        self._act_zone_mask: np.array = simulate_in.subzone.zone_mask

        assert (
            simulate_in.service_actions_nbr is not None
            or simulate_in.relocations_nbr is not None
        )
        self._service_actions_nbr: Optional[int] = simulate_in.service_actions_nbr
        self._relocations_nbr: Optional[int] = simulate_in.relocations_nbr
        self._weeks_of_simulations: int = simulate_in.config["weeks_of_simulations"]
        self._tank_capacity = simulate_in.config["tank_capacity"]
        self._fuel_usage: float = simulate_in.config["fuel_usage"]
        self._refueling_start_level: float = simulate_in.config["refueling_start_level"]
        self._stop_renting_fuel_level: float = simulate_in.config[
            "stop_renting_fuel_level"
        ]

        self._simulation_day = 0
        self._array_hour = 0
        self._simulation_hour = 0

        self.simulate_out: SimulateOut = None

    def simulate_weeks(self) -> None:
        """
        Simulate all weeks.
        """

        for day in range(0, self._weeks_of_simulations * 7):
            if day % 50 == 0:
                log.info(f"Simulate day {day}. of {self._weeks_of_simulations * 7}.")
                print(f"Simulate day {day}. of {self._weeks_of_simulations * 7}.")
            self._simulation_day = day
            self._array_hour = day % self._base_data.days * self._base_data.hours
            self._simulation_hour = day * self._base_data.hours
            self._simulate_day()
            self._refuel_and_relocate()
        self.simulate_out = SimulateOut(self._board.vehicles_all())

    def _simulate_day(self) -> None:
        """
        Simulate single day.
        """

        # for every hour
        for h in range(self._base_data.hours):
            ah = self._array_hour + h
            sh = self._simulation_hour + h
            rents = []

            # build rents
            # for every cell with vehicles
            for src_cell_idx in self._board.occupied_cells():
                cell_vehicles = self._board.vehicles_in_cell(src_cell_idx)
                cell_vehicles = [
                    v for v in cell_vehicles if v.fuel > self._stop_renting_fuel_level
                ]
                random.shuffle(cell_vehicles)
                cell_vehicles_idx = len(cell_vehicles)

                # for each demand in the cell
                actual_demand = Simulator._demand_integer(
                    self._base_data.rent_demand[ah, src_cell_idx]
                )
                while actual_demand > 0 and cell_vehicles_idx > 0:
                    actual_demand -= 1

                    # find destination cell
                    direction_sum = self._base_data.rent_direction[
                        ah, src_cell_idx
                    ].sum()
                    assert 0.99999 < direction_sum < 1.00001, str(
                        f"For hour {ah} src_cell {src_cell_idx} direction sum is {direction_sum}"
                    )
                    dst_cell_idx = Simulator._dst_cell_idx(
                        self._base_data.rent_direction[ah, src_cell_idx]
                    )

                    if self._act_zone_mask[dst_cell_idx]:
                        # rent vehicle
                        cell_vehicles_idx -= 1
                        rents.append([cell_vehicles[cell_vehicles_idx], dst_cell_idx])

            # apply rents
            rent_time_cc = self._rent_time_at(
                self._array_hour // self._base_data.hours, h
            )

            for rent in rents:
                vehicle, stop_cell_idx = rent
                start_cell_idx = vehicle.cell_idx

                rent_distance = self._base_data.route_distance[
                    start_cell_idx, stop_cell_idx
                ]
                rent_time = rent_time_cc[start_cell_idx, stop_cell_idx]
                rent_fuel = self._fuel_usage * rent_distance / 100_000

                self._board.move(vehicle, stop_cell_idx)
                vehicle.fuel -= rent_fuel
                rent = Rent(
                    sh,
                    vehicle.idx,
                    start_cell_idx,
                    stop_cell_idx,
                    rent_distance,
                    rent_time,
                )
                vehicle.history.append(rent)

    @staticmethod
    def _demand_integer(demand_float: float) -> int:
        """
        Get demand as integer. Fractional part of the demand is a probability of addition 1.

        :param demand_float: demand expected value

        :return: demand as integer
        """

        integer = int(demand_float)
        fractional = demand_float - integer
        p = np.array([1 - fractional, fractional])
        p /= p.sum()  # normalize array to avoid NaN in case of rounding errors
        return integer + np.random.choice([0, 1], p=p)

    @staticmethod
    def _dst_cell_idx(prob_c: np.ndarray) -> int:
        """
        Get destination cell index.

        :param prob_c: destination probability array

        :return: destination cell index
        """

        cells_nbr = prob_c.shape[0]
        return np.random.choice(cells_nbr, p=prob_c)

    def _rent_time_at(self, day: int, hour: int):
        """
        Computes rent time array at specific day and hour

        :param day: Day of the week
        :type day: int

        :param hour: Hour of the day
        :type hour: int

        :return: Rent time array at specific day and hour
        :rtype: numpy.ndarray, shape=(C, C)
        """

        rent_time_cc = (
            self._base_data.route_time * self._base_data.time_traffic_factor[day, hour]
        )
        return rent_time_cc

    def _refuel_and_relocate(self):
        """
        Performing refuel and relocation service actions. There are four cases differing in the number of individual
        types of simulated actions.
        """

        if self._service_actions_nbr is not None and self._relocations_nbr is not None:
            # service_actions_nbr is refueling number and relocations_nbr is relocations number
            refuelings_nbr = self._service_actions_nbr
            self._refuel(refuelings_nbr)
            relocations_nbr = self._relocations_nbr
            self._relocate(relocations_nbr)

        elif self._service_actions_nbr is None and self._relocations_nbr is not None:
            # refueling as many as needed: relocations_nbr is set but service_actions_nbr is not
            refuelings_nbr = len(self._board.vehicles_all())
            self._refuel(refuelings_nbr)
            relocations_nbr = self._relocations_nbr
            self._relocate(relocations_nbr)

        elif self._service_actions_nbr is not None and self._relocations_nbr is None:
            # service_actions_nbr is sum of refueling actions and relocations, refueling has priority
            refuelings_nbr = self._service_actions_nbr
            done_refuel_nbr = self._refuel(refuelings_nbr)
            relocations_nbr = self._service_actions_nbr - done_refuel_nbr
            self._relocate(relocations_nbr)

        else:
            raise TypeError(
                "At least one of: self.service_actions_nbr or self.relocations_nbr has to be an integer"
            )

    def _refuel(self, max_refuels: int) -> None:
        """
        Refuel vehicles.

        :param max_refuels: maximum number of vehicles to refuel
        """

        sh = self._simulation_hour + self._base_data.hours
        vehicles = self._board.vehicles_all()
        vehicles.sort(key=lambda v: v.fuel)
        count = 0
        for v in vehicles:
            if v.fuel < self._refueling_start_level:
                start_fuel = v.fuel
                stop_fuel = self._tank_capacity
                distance_to_station = self._base_data.distance_to_station[v.cell_idx]
                v.history.append(
                    Refuel(
                        sh,
                        v.idx,
                        start_fuel,
                        stop_fuel,
                        distance_to_station,
                        v.cell_idx,
                    )
                )
                v.fuel = stop_fuel
                count += 1
                if count >= max_refuels:
                    break
        return count

    def _relocate(self, limit):
        """
        Relocate vehicles to best cells.

        :param limit: maximum number of vehicles to relocate
        """

        sh = self._simulation_hour + self._base_data.hours
        zone_cells_list = list(np.where(self._act_zone_mask)[0])
        demand_per_vehicle_c = self._build_average_demand_per_vehicle_in_cell()
        cell_quality_order_dpv = np.argsort(-demand_per_vehicle_c)
        cell_quality_order_dpv = [
            c for c in cell_quality_order_dpv if c in zone_cells_list
        ]
        vehicles = self._board.vehicles_all()
        vehicles.sort(key=lambda v: demand_per_vehicle_c[v.cell_idx])
        for i, v in enumerate(vehicles):
            if i >= limit:
                break
            else:
                src_cell_idx = v.cell_idx
                zone_cells_number = len(self._cell_quality.zone_order_i)
                dst_cell_limit = int(min(limit, zone_cells_number))
                if dst_cell_limit > 0:
                    dst_cell_idx = cell_quality_order_dpv[i % dst_cell_limit]
                    assert dst_cell_idx in zone_cells_list
                    distance = self._base_data.here_distance[src_cell_idx, dst_cell_idx]
                    v.history.append(
                        Relocation(sh, v.idx, src_cell_idx, dst_cell_idx, distance)
                    )
                    self._board.move(v, dst_cell_idx)

    def _build_average_demand_per_vehicle_in_cell(self) -> np.ndarray:
        """
        Build array of demand per vehicle indexed by cell index.

        :return: Array of demand per vehicle indexed by cell index
        """

        vehicles = self._board.vehicles_all()
        cells_nbr = self._cell_quality.quality_c.shape[0]
        vehicles_nbr_c = np.zeros(cells_nbr, dtype=np.int32)
        demand_per_vehicle_c = self._cell_quality.quality_c

        for v in vehicles:
            vehicles_nbr_c[v.cell_idx] += 1

        for cell in range(cells_nbr):
            if vehicles_nbr_c[cell] != 0:
                demand_per_vehicle_c[cell] = self._cell_quality.quality_c[cell] / (
                    1 + vehicles_nbr_c[cell]
                )
        return demand_per_vehicle_c
