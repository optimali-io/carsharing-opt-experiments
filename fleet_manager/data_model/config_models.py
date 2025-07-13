from typing import List

from pydantic import BaseModel, Field, PositiveInt, condecimal



class GeneticConfiguration(BaseModel):
    """Configuration for genetic algorithm."""

    project_name: str = "Default project"
    population_size: int  # population size
    maximum_time: int  # maximum execution time in seconds
    maximum_no_improvement: int  # maximum number of generation without improvement
    maximum_generation_number: int  # maximum number of generation
    maximum_tune_generation_number: int  # maximum number of generation during tune mode

    island_population: int  # maximum population size on each island (used in island parallel mode)
    migration_interval: int  # number of generation between migrations (used in island parallel mode)
    n_neighbors: int  # number of neighbors used in similarity graph building
    tolerance: float  # how small should kneighbor graph's laplacian eigenvalues be to be considered as zero

    chromosome_deviation: int  # maximum deviation of chromosome in percentages
    tournament_samples: int  # number of samples in tournament selection

    probability_gene_mutation: float  # probability of gene mutation
    probability_action_change: float  # probability of action change
    # complement probability is destination change

    mutation_types: List[str]
    weight_mutation_types: List[int]  # weights for gene mutation types

    tuned_mutation_types: List[str]
    tuned_weight_mutation_types: List[int]  # weights for gene mutation types


class ZoneDataConfig(BaseModel):
    """ZoneDataConfig model containing configuration parameters of FM's zone data"""

    action_duration_1: PositiveInt = Field(
        default=45, description="Average action duration in minutes of 1-person service team."
    )
    action_duration_2: PositiveInt = Field(
        default=35, description="Average action duration in minutes of 2-person service team."
    )
    price_per_action: bool = Field(
        default=False, description="If True, evaluation function counts cost per action."
    )
    begin_charge_time: PositiveInt = Field(
        default=360, description="Additional average time in seconds for begin charging vehicle."
    )
    end_charge_time: PositiveInt = Field(
        default=480, description="Additional average time in seconds for end charging vehicle."
    )

    relocation_max_distance: PositiveInt = Field(
        default=5000, description="Maximum relocation distance in meters."
    )
    revenue_zero_range: PositiveInt = Field(
        default=70, description="Range in km below which the vehicle is inactive and cannot be rented."
    )
    revenue_reduction_range: PositiveInt = Field(
        default=150, description="Range in km below which revenue is reduced."
    )
    revenue_start_reduction_time: PositiveInt = Field(
        default=1440, description="Wait time in minutes when revenue reduction is started."
    )
    revenue_end_reduction_time: PositiveInt = Field(
        default=2880, description="Wait time in minutes when revenue reduction drops to zero."
    )
    allowed_cells_demand_percentile: PositiveInt = Field(
        default=75, description="Percentile from which allowed cells are taken."
    )
    minimum_cells_in_range: PositiveInt = Field(
        default=12, description="Minimum number of available cells for relocation."
    )

    service_exceeded_time_penalty: float = Field(
        default=0.00,
        description="Penalty cost in adopted currency (e.g. PLN) for 1 second of additional service work.",
    )
    service_refuel_bonus: float = Field(
        default=30, description="Bonus for each refuel."
    )
    service_wait_time_bonus: float = Field(
        default=15, description="Bonus for each relocation."
    )
    blur_factor: float = Field(
        default=0.00, description="Blur factor for demand modification."
    )
    blur_distance: float = Field(
        default=0.00, description="Blur distance for demand modification [m]."
    )

class Zone(BaseModel):
    id: str
    cell_ids: list[str]

class ScienceConfig(BaseModel):
    """Configuration for science module."""

    genetic_config: GeneticConfiguration = Field(
        default_factory=GeneticConfiguration, description="Configuration for genetic algorithm."
    )
    zone_data_config: ZoneDataConfig = Field(
        default_factory=ZoneDataConfig, description="Configuration for zone data."
    )
    zone: Zone