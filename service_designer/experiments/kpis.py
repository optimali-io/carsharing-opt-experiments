import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from service_designer.data.base_data import BaseData
from service_designer.simulator.simulator_daybyday import Refuel, Relocation, Rent, Vehicle

log = logging.getLogger("service_designer")


class PriceList:
    """
    Defines service costs and rent prices as arguments

    :param config: rent and service price list collected as dictionary
    """

    def __init__(self, config: Dict):
        #: Revenue per start
        self.price_per_rent_start: float = float(config["price_per_rent_start"])
        #: Revenue per km
        self.price_per_m: float = float(config["price_per_km"]) / 1000
        #: Revenue per min
        self.price_per_s: float = float(config["price_per_minute"]) / 60
        #: Cost of service action ride per km
        self.cost_per_action_m: float = float(config["cost_per_action_km"]) / 1000
        #: Cost of one relocation
        self.cost_per_relocation: float = float(config["cost_per_relocation"])
        #: Cost of one refueling action
        self.cost_per_refueling: float = float(config["cost_per_refueling"])
        #: The arrival/journey factor of a two-person service
        self.service_factor: float = float(config["service_factor"])


class Kpis:
    """
    Class responsible for processing simulation data and calculating the Key Performance Indicators
    of simulated carsharing service.

    :param vehicles: list of vehicles with their whole history of simulated actions
    :param base_data: BaseData object of zone spacial information grid, distances, etc.
    :param price_list: PriceList object of service costs and rent prices as arguments
    """

    def __init__(self, vehicles: List[Vehicle], base_data: BaseData, price_list: PriceList, config: Dict):
        self.vehicles: List[Vehicle] = vehicles
        self.base_data: BaseData = base_data
        self.price_list: PriceList = price_list
        self.stations: pd.DataFrame = self.base_data.stations

        self.days_in_week: int = self.base_data.days
        self.hours_in_day: int = self.base_data.hours

        self.weeks_of_simulations = config["weeks_of_simulations"]
        self.exclude_first_weeks = config["exclude_first_weeks"]

        self.total_rent_number: int = 0
        self.total_rent_revenue: float = 0.0
        self.total_rent_distance: float = 0.0  # m
        self.total_rent_time: float = 0.0  # s

        self.total_relocations_number: int = 0
        self.total_relocations_cost: float = 0.0
        self.total_relocations_distance: float = 0.0  # m

        self.total_refueling_number: int = 0
        self.total_refueling_cost: float = 0.0
        self.total_refueling_distance: float = 0.0  # m
        self.refueling_source_count: np.array = np.zeros(shape=self.base_data.cells)

        self.total_profit: float = None
        self.total_demand: float = None
        self.satisfied_demand: float = None

        self.day_rent_number: int = None
        self.day_rent_revenue: float = None
        self.day_rent_distance: float = None
        self.day_rent_time: float = None

        self.day_relocation_number: float = None
        self.day_relocation_cost: float = None
        self.day_relocation_distance: float = None

        self.day_refuel_number: float = None
        self.day_refuel_cost: float = None
        self.day_refuel_distance: float = None

        self.day_profit: float = None
        self.day_demand: float = None

        self.mean_rent_revenue: float = None
        self.mean_rent_distance: float = None
        self.mean_rent_time: float = None
        self.mean_rent_profit: float = None

        self.day_vehicle_rent_number: float = None
        self.day_vehicle_rent_revenue: float = None
        self.day_vehicle_rent_distance: float = None
        self.day_vehicle_rent_time: float = None
        self.day_vehicle_profit: float = None

    def calculate_and_clear_working_data(self):
        """Calculate KPIS and clear working data."""
        self.calculate_total_kpis()
        self.calculate_daily_kpis()
        self.calculate_daily_kpis_per_vehicle()
        self.calculate_kpis_per_rent()
        self.clear_working_data()

    def clear_working_data(self):
        """Reduce memory copy when Kpis returns as result from subprocess."""
        self.vehicles = None
        self.base_data = None

    def _calculate_total_demand(self):
        """Return the value of summarized demand for typical week from all zone cells."""
        self.total_demand = np.sum(self.base_data.rent_demand) * (self.weeks_of_simulations - self.exclude_first_weeks)

    def _calculate_raw_kpis(self):
        """Calculate raw total KPIs without financial statistics (excluding calculations based on price lists)."""
        calc_start_hour = self.exclude_first_weeks * self.days_in_week * self.hours_in_day
        for _, vehicle in enumerate(self.vehicles):
            for action in vehicle.history:
                if action.hour > calc_start_hour:
                    if isinstance(action, Rent):
                        self.total_rent_number += 1
                        self.total_rent_distance += action.distance
                        self.total_rent_time += action.time

                    elif isinstance(action, Relocation):
                        self.total_relocations_number += 1
                        self.total_relocations_distance += action.distance

                    elif isinstance(action, Refuel):
                        self.total_refueling_number += 1
                        self.total_refueling_distance += action.distance_to_station
                        self.refueling_source_count[action.source_cell_idx] += 1

        assert sum(len(v.history) for v in self.vehicles) >= sum(
            [self.total_rent_number, self.total_refueling_number, self.total_relocations_number]
        )
        assert self.total_refueling_number == sum(self.refueling_source_count)

    def calculate_total_kpis(self):
        """Calculate total KPIs for simulation period excluding a few first weeks."""

        self._calculate_total_demand()
        self._calculate_raw_kpis()

        self.total_rent_revenue += (
            (self.price_list.price_per_rent_start * self.total_rent_number)
            + (self.price_list.price_per_m * self.total_rent_distance)
            + (self.price_list.price_per_s * self.total_rent_time)
        )
        self.total_relocations_cost = (self.price_list.cost_per_relocation * self.total_relocations_number) + (
            self.price_list.service_factor * self.price_list.cost_per_action_m * self.total_relocations_distance
        )
        self.total_refueling_cost = (self.price_list.cost_per_refueling * self.total_refueling_number) + (
            self.price_list.service_factor * self.price_list.cost_per_action_m * self.total_refueling_distance
        )

        self.total_profit = self.total_rent_revenue - (self.total_refueling_cost + self.total_relocations_cost)
        self.satisfied_demand = self.total_rent_number / self.total_demand

        station_count = len(self.stations.index)
        station_occupancy = {}
        for station_id in range(station_count):
            cell_ids = eval(self.stations["cell_ids"][station_id])
            station_occupancy[station_id] = sum(self.refueling_source_count[cell_ids])

        self.stations["occupancy"] = pd.Series(station_occupancy)

        assert sum(self.stations["occupancy"]) == self.total_refueling_number

        log.info(
            f"Annual balance for {len(self.vehicles)} vehicles and {self.base_data.cells} cells is {self.total_profit}\n"
            f"\tRents number: {self.total_rent_number}\n"
            f"\tRevenue: {self.total_rent_revenue}\n "
            f"\tRelocation cost: {self.total_relocations_cost}\n "
            f"\tRefueling cost: {self.total_refueling_cost}\n"
            f"\tProfit: {self.total_profit}\n"
            f"\tSatisfied demand: {self.satisfied_demand}"
        )

    def calculate_daily_kpis(self):
        """Calculate daily KPIs for simulation period excluding a few first weeks."""
        significant_simulation_days = (self.weeks_of_simulations - self.exclude_first_weeks) * self.days_in_week

        self.day_rent_number = round(self.total_rent_number / significant_simulation_days, 2)
        self.day_rent_revenue = round(self.total_rent_revenue / significant_simulation_days, 2)
        self.day_rent_distance = round(self.total_rent_distance / significant_simulation_days, 2)
        self.day_rent_time = round(self.total_rent_time / significant_simulation_days, 2)

        self.day_refuel_number = round(self.total_refueling_number / significant_simulation_days, 2)
        self.day_refuel_cost = round(self.total_refueling_cost / significant_simulation_days, 2)
        self.day_refuel_distance = round(self.total_refueling_distance / significant_simulation_days, 2)

        self.day_relocation_number = round(self.total_relocations_number / significant_simulation_days, 2)
        self.day_relocation_cost = round(self.total_relocations_cost / significant_simulation_days, 2)
        self.day_relocation_distance = round(self.total_relocations_distance / significant_simulation_days, 2)

        self.day_profit = round(self.total_profit / significant_simulation_days, 2)
        self.day_demand = round(self.total_demand / significant_simulation_days, 2)

    def calculate_daily_kpis_per_vehicle(self):
        """Calculate daily KPIs per Vehicle."""
        significant_simulation_days = (self.weeks_of_simulations - self.exclude_first_weeks) * self.days_in_week
        vehicle_number = len(self.vehicles)

        self.day_vehicle_rent_number = round(self.total_rent_number / significant_simulation_days / vehicle_number, 2)
        self.day_vehicle_rent_revenue = round(self.total_rent_revenue / significant_simulation_days / vehicle_number, 2)
        self.day_vehicle_rent_distance = round(
            self.total_rent_distance / significant_simulation_days / vehicle_number, 2
        )
        self.day_vehicle_rent_time = round(self.total_rent_time / significant_simulation_days / vehicle_number, 2)
        self.day_vehicle_profit = round(self.total_profit / significant_simulation_days / vehicle_number, 2)

    def calculate_kpis_per_rent(self):
        """Calculate mean KPIs for rent."""

        self.mean_rent_revenue = self.total_rent_revenue / self.total_rent_number
        self.mean_rent_distance = self.total_rent_distance / self.total_rent_number
        self.mean_rent_time = self.total_rent_time / self.total_rent_number
        self.mean_rent_profit = self.total_profit / self.total_rent_number

    def evaluate_result(self):
        """Simple evaluation function. Calculate the score of simulated carsharing service based on KPIs."""
        return round(1 - 1_000_000.0 / self.total_rent_revenue, 4)