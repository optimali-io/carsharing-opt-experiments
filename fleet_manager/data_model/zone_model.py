import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import Field, BaseModel
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform

from fleet_manager.data_model.config_models import ScienceConfig

log = logging.getLogger(__name__)


class VehicleModelName(str, Enum):
    yaris = "yaris"

class FuelType(str, Enum):
    PETROL = 1
    ELECTRIC = 2

class VehicleModel(BaseModel):
    average_fuel_consumption: float # l/100km
    tank_capacity: float
    fuel_type: FuelType

class VehicleModels(BaseModel):
    models: dict[str, VehicleModel]

class ActionType:
    """Type of a service action."""

    N = 0  # Non action
    F = 1  # reFueling
    R = 2  # Relocation
    P = 3  # Plugging in
    U = 4  # Unplugging
    C = 5  # Charging = (Plugging in + Unplugging)


class Cell:
    """
    Representation of a single cell in a Zone.
    """

    def __init__(self, cell_id: int, in_zone: bool, longitude: float = 0, latitude: float = 0):
        self.cell_id: int = cell_id
        self.base_vehicles: List[Vehicle] = []  #
        self.base_vehicle_number: int = 0  #
        self.base_revenue: float = 0
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.involved: bool = False
        self.vehicles: List[Vehicle] = []
        self.vehicle_number: int = 0


class Vehicle:
    """Representation of a vehicle."""

    def __init__(
        self,
        vehicle_index: int,
        backend_vehicle_id: int,
        cell_id: int,
        max_range: int,
        range: int,
        wait_time: int,
        latitude: float = 0,
        longitude: float = 0,
        drive: int = 1,
        power_station_id: int = -1,
    ):
        self.vehicle_index: int = vehicle_index
        self.backend_vehicle_id: int = backend_vehicle_id
        self.base_cell_id: int = cell_id
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.max_range: int = max_range
        self.range: int = range
        self.wait_time: int = wait_time  # minutes
        self.drive: int = drive  # 1 - petrol, 2 - electric
        self.power_station_id: int = power_station_id
        self.charging_factor: float = 1.8  # in additional range in km per minute of charging
        self.base_revenue_range_factor: float = 1.0
        self.base_revenue_time_factor: float = 1.0
        self.revenue_range_factor: float = 1.0
        self.revenue_time_factor: float = 1.0
        self.start_charging_time: int = -1
        self.end_charging_time: int = -1
        self.allowed_actions: List[int] = []

        self.moved: bool = False
        self.current_cell_id: int = self.base_cell_id
        self.current_wait_time: int = self.wait_time
        self.current_range: int = self.range

    def set_allowed_actions(self):
        """
        Set allowed action types for a vehicle. For example if a vehicle has a full fuel tank it should not be refuelled.
        """
        if self.power_station_id == -1:
            self.allowed_actions.append(ActionType.R)
        else:
            self.allowed_actions.append(ActionType.U)
        if self.base_revenue_range_factor < 1:
            if self.drive == 1:
                self.allowed_actions.append(ActionType.F)
            else:
                if self.power_station_id == -1:
                    self.allowed_actions.append(ActionType.P)
                    self.allowed_actions.append(ActionType.C)
                else:
                    self.allowed_actions.append(ActionType.U)

    def __str__(self):
        return "V: {0}, S: {1}, BRF: {2}, BRT: {3}, RF: {4}, RT: {5}: R: {6}\n".format(
            self.vehicle_index,
            self.base_cell_id,
            self.base_revenue_range_factor,
            self.base_revenue_time_factor,
            self.revenue_range_factor,
            self.revenue_time_factor,
            self.range,
        )


class PowerStation:
    """Representation of a charging station."""

    def __init__(self, power_station_id: int, cell_id: int, capacity: int, latitude: float, longitude: float):
        self.power_station_id: int = power_station_id
        self.cell_id: int = cell_id
        self.capacity: int = capacity
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.base_charging_vehicles: List[dict] = []
        self.charging_vehicles: List[dict] = []


class GasStation:
    """Representation of a gas station."""

    def __init__(
        self, gas_station_id: int, gas_station_name: str, latitude: float, longitude: float, cell_id: int, address: str
    ):
        self.gas_station_id: int = gas_station_id
        self.gas_station_name: str = gas_station_name
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.cell_id: int = cell_id
        self.address: str = address


class ServiceTeam(BaseModel):
    """Representation of a service team"""
    service_team_id: int
    service_team_name: str
    service_team_kind: int
    planned_time_work_minutes: int
    time_cost_per_second: float
    distance_cost_per_km: float
    start_cell_id: int = -1
    end_cell_id: int = -1
    refuel_time_seconds: int = 1080
    relocation_time_seconds: int = 480
    price_per_relocation: float | None = None
    price_per_refuelling: float | None = None



class ZoneData(BaseModel):
    """Class aggregates zone state and configuration for optimisation algorithm."""

    class Config:
        arbitrary_types_allowed = True

    zone_id: str
    client_cell_ids: List[str]
    base_revenue: float = 0
    cells: List[Cell] = Field(default_factory=list)
    cell_number: int
    zone_cell: List[int] = Field(default_factory=list)
    snap_point: List[Optional[dict]] = Field(default_factory=list)
    fleet: List[Vehicle] = Field(default_factory=list)
    fleet_ids_to_refuel: List[int] = Field(default_factory=list)
    service_team: List[ServiceTeam] = Field(default_factory=list)
    service_team_id_by_index_dict: Dict[int, int] = Field(default_factory=dict)
    gas_station: List[GasStation] = Field(default_factory=list)
    power_station: List[PowerStation] = Field(default_factory=list)
    allowed_cells: List[Optional[List[int]]] = Field(default_factory=list)
    allowed_cells_ps: List[Optional[List[int]]] = Field(default_factory=list)
    parking_blacklist_cells: List[Optional[List[int]]] = Field(default_factory=list)

    demand: Optional[np.core.multiarray] = None
    distance_cell_cell: Optional[np.core.multiarray] = None
    distance_cell_station_cell: Optional[np.core.multiarray] = None
    id_cell_station_cell: Optional[np.core.multiarray] = None
    time_cell_cell: Optional[np.core.multiarray] = None
    time_cell_station_cell: Optional[np.core.multiarray] = None
    revenue: Optional[np.core.multiarray] = None

    action_duration_1: int
    action_duration_2: int
    price_per_action: bool
    begin_charge_time: int
    end_charge_time: int
    service_exceeded_time_penalty: float
    service_refuel_bonus: float
    service_wait_time_bonus: float
    allowed_cells_demand_percentile: int
    revenue_zero_range: int
    revenue_reduction_range: int
    revenue_start_reduction_time: int
    revenue_end_reduction_time: int
    minimum_cells_in_range: int
    geocoding: bool = True
    relocation_max_distance: int

    @classmethod
    def from_science_config(cls, science_config: ScienceConfig) -> "ZoneData":
        """
        Creates ZoneData object from ScienceConfig.
        :param science_config: configuration
        :return: ZoneData object
        """
        return cls(
            zone_id=science_config.zone.id,
            client_cell_ids=science_config.zone.cell_ids,
            cell_number=len(science_config.zone.cell_ids),
            action_duration_1=science_config.zone_data_config.action_duration_1,
            action_duration_2=science_config.zone_data_config.action_duration_2,
            price_per_action=science_config.zone_data_config.price_per_action,
            begin_charge_time=science_config.zone_data_config.begin_charge_time,
            end_charge_time=science_config.zone_data_config.end_charge_time,
            service_exceeded_time_penalty=float(science_config.zone_data_config.service_exceeded_time_penalty),
            service_refuel_bonus=float(science_config.zone_data_config.service_refuel_bonus),
            service_wait_time_bonus=float(science_config.zone_data_config.service_wait_time_bonus),
            allowed_cells_demand_percentile=science_config.zone_data_config.allowed_cells_demand_percentile,
            revenue_zero_range=science_config.zone_data_config.revenue_zero_range,
            revenue_reduction_range=science_config.zone_data_config.revenue_reduction_range,
            revenue_start_reduction_time=science_config.zone_data_config.revenue_start_reduction_time,
            revenue_end_reduction_time=science_config.zone_data_config.revenue_end_reduction_time,
            minimum_cells_in_range=science_config.zone_data_config.minimum_cells_in_range,
            relocation_max_distance=science_config.zone_data_config.relocation_max_distance,
        )


    def init_zone_data(self):
        """
        Method initializes factors, revenues and values in ZoneData.
        """
        self._set_allowed_cells()
        self.initialize_cells()
        self._initialize_fleet()
        self._initialize_service_teams()
        self._compute_base_revenue()

    def _set_allowed_cells(self) -> None:
        """
        Sets possible destinations for every cell in ZoneData.
        Cells that are too far or are blacklisted can't be destination.
        """

        self.allowed_cells = [None for _ in range(self.cell_number)]
        self.allowed_cells_ps = [None for _ in range(self.cell_number)]
        radius: int = self.relocation_max_distance

        src_cell_id = 0
        while src_cell_id < self.cell_number:
            cells_in_range: List[int] = []
            demand_in_range: List[int] = []
            for dst_cell_id in self.zone_cell:
                if (
                    dst_cell_id not in self.parking_blacklist_cells
                    and self.distance_cell_cell[src_cell_id, dst_cell_id] <= radius
                ):
                    cells_in_range.append(dst_cell_id)
                    demand_in_range.append(self.revenue[dst_cell_id, 0])

            din: int = len(demand_in_range)

            if din < self.minimum_cells_in_range:
                radius *= 1.2
                if radius > 100000:
                    raise ValueError(f"Can not find destinations for cell {src_cell_id}")
            else:
                demand_threshold = np.percentile(demand_in_range, self.allowed_cells_demand_percentile)
                high_demand_cells_indices = np.argwhere(demand_in_range >= demand_threshold).reshape(-1).tolist()
                self.allowed_cells[src_cell_id] = [cells_in_range[idx] for idx in high_demand_cells_indices]
                src_cell_id += 1
                radius = self.relocation_max_distance
        src_cell_id = 0
        while src_cell_id < self.cell_number:
            cells_in_range = []
            demand_in_range = []
            for dst_cell_id in self.zone_cell:
                if (
                    dst_cell_id not in self.parking_blacklist_cells
                    and self.distance_cell_station_cell[src_cell_id, dst_cell_id] <= radius
                ):
                    cells_in_range.append(dst_cell_id)
                    demand_in_range.append(self.revenue[dst_cell_id, 0])

            din = len(demand_in_range)

            if din < self.minimum_cells_in_range:
                radius *= 1.2
            else:
                demand_threshold = np.percentile(demand_in_range, self.allowed_cells_demand_percentile)
                high_demand_cells_indices = np.argwhere(demand_in_range >= demand_threshold).reshape(-1).tolist()
                self.allowed_cells_ps[src_cell_id] = [cells_in_range[idx] for idx in high_demand_cells_indices]
                src_cell_id += 1
                radius = self.relocation_max_distance

    def initialize_cells(self) -> None:
        """Creates list of Cell objects and sets their 'in_zone' value."""
        for i, cell_id in enumerate(self.client_cell_ids):
            lon, lat = cell_id.split("-")
            self.cells.append(Cell(i, True, longitude=float(lon), latitude=float(lat)))

    def _initialize_fleet(self) -> None:
        """Sets allowed actions, range and time factors."""
        for vehicle in self.fleet:
            vehicle.base_revenue_range_factor = self.compute_range_factor(vehicle.range)
            if vehicle.range < self.revenue_reduction_range and vehicle.drive == 1:
                self.fleet_ids_to_refuel.append(vehicle.vehicle_index)
            self._compute_base_time_factor(vehicle)
            self.cells[vehicle.base_cell_id].base_vehicles.append(vehicle)
            self.cells[vehicle.base_cell_id].base_vehicle_number += 1
            if vehicle.power_station_id != -1:
                charging_vehicle: dict = dict(vehicle_id=vehicle.vehicle_index, action_time=0, action_type=ActionType.P)
                self.power_station[vehicle.power_station_id].base_charging_vehicles.append(charging_vehicle)
            vehicle.revenue_range_factor = vehicle.base_revenue_range_factor
            vehicle.revenue_time_factor = vehicle.base_revenue_time_factor
            vehicle.start_charging_time = -1
            vehicle.end_charging_time = -1
            vehicle.set_allowed_actions()

    def _compute_base_revenue(self) -> None:
        """Calculates revenue from each cell before optimisation."""
        for cell in self.cells:
            vehicle_number: int = cell.base_vehicle_number
            factors: float = 0
            if vehicle_number > 0:
                for vehicle in cell.base_vehicles:
                    if vehicle.base_revenue_range_factor == 0 or vehicle.base_revenue_time_factor == 0:
                        vehicle_number -= 1
                    factors += vehicle.base_revenue_range_factor * vehicle.base_revenue_time_factor
                if vehicle_number > 10:
                    vehicle_number = 10
                if vehicle_number > 0:
                    revenue_per_vehicle: float = self.revenue[cell.cell_id, vehicle_number - 1] / vehicle_number
                    if vehicle_number == 1:
                        cell.base_revenue = 0.95 * revenue_per_vehicle * factors
                    elif vehicle_number == 2:
                        cell.base_revenue = 0.9 * revenue_per_vehicle * factors
                    else:
                        cell.base_revenue = revenue_per_vehicle * factors
                else:
                    cell.base_revenue = 0
            self.base_revenue += cell.base_revenue

    def _compute_base_time_factor(self, vehicle: Vehicle) -> None:
        """Computes vehicle revenue reduction after given waiting time."""
        if self.price_per_action:
            if vehicle.wait_time > self.revenue_end_reduction_time:
                vehicle.base_revenue_time_factor = 0.0
            else:
                vehicle.base_revenue_time_factor = 1.0
        else:
            t1: int = self.revenue_start_reduction_time  # since this time revenue starts to drop
            t2: int = self.revenue_end_reduction_time  # since this time revenue reaches zero value
            if vehicle.wait_time > t2:
                vehicle.base_revenue_time_factor = 0.0
            elif vehicle.wait_time <= t1:
                vehicle.base_revenue_time_factor = 1.0
            else:
                k: float = 1.0 / ((t2 - t1) ** 2)
                vehicle.base_revenue_time_factor = k * (
                    -vehicle.wait_time ** 2 + 2 * t1 * vehicle.wait_time + t2 * (t2 - 2 * t1)
                )

    def compute_range_factor(self, range: int) -> float:
        """Computes vehicle revenue reduction based on its range."""
        if range < self.revenue_reduction_range:
            if range <= self.revenue_zero_range:
                return 0.0
            else:
                return (range / self.revenue_reduction_range) ** 3
        else:
            return 1.0

    def get_lat_lon(self, cell_id: int) -> Tuple[float, float]:
        """Get cell latitude, longitude coordinates.

        :param cell_id: cell identifier
        :type cell_id: int

        :return: cell latitude, longitude coordinates
        :rtype: Tuple[float, float]
        """
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell.latitude, cell.longitude
        raise KeyError(f"No cell for cell_id {cell_id}")

    def get_nearest_cell_id_for_point(self, latitude: float, longitude: float) -> int:
        """Get nearest cell id.

        :param latitude: latitude
        :type latitude: float

        :param longitude: longitude
        :type latitude: float

        :return: cell id
        :rtype: int
        """
        cell = self._get_nearest_cell_for_point(latitude, longitude)
        return cell.cell_id

    def _get_nearest_cell_for_point(self, latitude: float, longitude: float) -> Cell:
        """
        Return Cell which snapping point is nearest to provided coords.
        :param latitude: latitude wgs84, float
        :param longitude: longitude wgs84, float
        :return: nearest Cell
        """
        wgs84 = "epsg:4326"
        cs92 = "epsg:2180"

        transform_to_plane = Transformer.from_crs(wgs84, cs92).transform

        point_sphere = Point(longitude, latitude)
        point_plane = transform(transform_to_plane, point_sphere)

        nearest_cell: Cell = self.cells[0]
        cell_point_sphere = Point(nearest_cell.longitude, nearest_cell.latitude)
        cell_point_plane = transform(transform_to_plane, cell_point_sphere)
        nearest_cell_distance = point_plane.distance(cell_point_plane)

        for _, cell in enumerate(self.cells):
            cell_point_sphere = Point(cell.longitude, cell.latitude)
            cell_point_plane = transform(transform_to_plane, cell_point_sphere)
            distance = point_plane.distance(cell_point_plane)
            if distance < nearest_cell_distance:
                nearest_cell_distance = distance
                nearest_cell = cell
        return nearest_cell

    def _initialize_service_teams(self):
        """
        This function maps the service team id from api to index in service teams array. Using arrays with indices is
        preferred over dictionaries because lookups are much faster by an index than by a key.
        """
        for i, st in enumerate(self.service_team):
            self.service_team_id_by_index_dict[i] = st.service_team_id
            st.service_team_id = i

    def get_service_team_id_by_index(self, service_team_index):
        """Function return service team id (from backend) by index from ZoneData's service teams array."""
        return self.service_team_id_by_index_dict[service_team_index]
