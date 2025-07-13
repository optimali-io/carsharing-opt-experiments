from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

from config import settings


class Subzone(BaseModel):
    name: str
    base_zone_name: str
    cell_ids: List[str] = Field(default_factory=list)


class SubzonesGeneratorConfig(BaseModel):
    """Configuration for subzones generator"""

    name: str = Field(..., max_length=256)
    subzone_cells_origin: str = Field(
        default="by_demand",
        description="How to sort cells in subzones, by demand, by rents or by profit",
    )
    subzone_count: int = Field(
        default=10, ge=1, description="Number of subzones to be created"
    )
    minimum_cells_in_subzone: int = Field(
        default=100, ge=1, description="Minimum number of cells in subzone"
    )


class BaseZone(BaseModel):
    name: str
    timezone: str | None = None
    cell_ids: list[str]


class DataConfig(BaseModel):
    """Properties of experiment input data set and subzones generator"""

    name: str = Field(..., max_length=256)
    base_zone: BaseZone
    start_date: date
    end_date: date
    approximation_directions_start_date: Optional[date] = None
    approximation_directions_end_date: Optional[date] = None
    include_holidays: bool = Field(
        default=True,
        description="Including Rents of holidays and specific days into base data",
    )
    rent_max_distance: int = Field(default=100)
    blur_factor: float = Field(
        default=0.07, ge=0.00, description="Blur factor for demand modification"
    )
    blur_distance: float = Field(
        default=800.00, ge=0.00, description="Blur distance for demand modification [m]"
    )


class SimulatorConfig(BaseModel):
    # DayByDay Simulator properties
    #: How many weeks simulate
    weeks_of_simulations: int = 52
    exclude_first_weeks: int = 0
    #: Tank capacity of Vehicles
    tank_capacity: float = Field(default=36.00, ge=0.0)
    #: Mean fuel usage by Vehicle
    fuel_usage: float = Field(default=6.00, ge=0.0)
    #: When Vehicle needs to be refueled
    refueling_start_level: float = Field(default=6.00, ge=0.0)
    #: When Vehicle cannot be rented
    stop_renting_fuel_level: float = Field(default=6.00, ge=0.0)

class PriceListConfig(BaseModel):
    #: Initial price of rent
    price_per_rent_start: float = Field(default=0.00, ge=0.0)
    # Price in adopted currency (e.g. PLN) for 1 distance billing unit
    price_per_km: float = Field(default=0.80, ge=0.0)
    # Price in adopted currency (e.g. PLN) for 1 time billing unit
    price_per_minute: float = Field(default=0.50, ge=0.0)
    # Cost of service action in adopted currency (e.g. PLN) for 1 distance billing unit
    cost_per_action_km: float = Field(default=0.30, ge=0.0)
    #: The arrival/journey factor of a two-person service
    service_factor: float = Field(default=2.00, ge=0.0)
    # Cost of one relocation in adopted currency (e.g. PLN)
    cost_per_relocation: float = Field(default=15.00, ge=0.0)
    # Cost of one refueling in adopted currency (e.g. PLN)
    cost_per_refueling: float = Field(default=35.00, ge=0.0)


class ExperimentConfig(BaseModel):
    """Properties of experiment, its identifiers and related input data set."""


    #: Name of experiment
    name: str = Field(..., max_length=256)
    #: Experiment's type
    type: str = Field(default="evaluation", max_length=24)

    #: Foreign key to input data set
    data_config: DataConfig

    #: Check while want to generate subzones automatically
    generate_subzones: bool = False
    #: Foreign key to subzones generator
    subzones_generator_config: Optional[SubzonesGeneratorConfig] = None

    #: Set of subzones taken to experiment
    subzones_set: List[Subzone] | None = None
    #: Set of vehicles number
    vehicle_count_set: List[int] = Field(default_factory=lambda: [100])
    #: Set of service actions number
    service_action_count_set: Optional[List[int]] = Field(default_factory=lambda: [15])
    #: Percent of relocations number in service actions
    relocation_actions_proportion_set: Optional[List[float]] = Field(
        default=None, ge=0.0, le=1.0
    )

    simulator_config: SimulatorConfig

    price_list_config: PriceListConfig

    #: Simulations mode of SimulatorFit experiment - to calculate statistics of simulations
    simulations_mode: bool = True
    #: Mode of SimulatorFit experiment to calculate statistics of real data
    real_statistics_mode: bool = False

    def get_experiment_zone_id(self) -> str:
        """
        Get experiment zone id.

        :returns: experiment zone id
        """
        return f"{settings.SDESIGNER_EXP_ZONES_DIR_NAME}/{self.data_config.name}"
