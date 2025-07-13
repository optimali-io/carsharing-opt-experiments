import logging
from typing import Dict

from pydantic import BaseModel, Field

from service_designer.experiments.kpis import PriceList, Kpis
from service_designer.simulator.simulator_daybyday import SimulateIn

log = logging.getLogger("service_designer")


class ParallelSimulateIn:
    """
    Input params of SimulatorDayByDay and KPIs executed for experiments on redis queues.

    :param simulate_in: SimulateIn configuration parameters
    :param price_list: PriceList configuration parameters
    """

    def __init__(self, simulate_in: SimulateIn, price_list: PriceList):
        self.simulate_in: SimulateIn = simulate_in
        self.price_list: PriceList = price_list


class SimulationResult(BaseModel):
    experiment_id: str = Field(..., description="ID of the experiment")
    subzone_id: str = Field(..., description="ID of the subzone")
    vehicle_count: int = Field(..., description="Number of vehicles")
    action_count: int = Field(..., description="Number of service actions")
    relocation_actions_proportion: float = Field(..., description="Proportion of relocation actions")
    rent_count: Dict[str, int] = Field(..., description="Total rent count")
    relocation_count: Dict[str, int] = Field(..., description="Total relocation count")
    refueling_count: Dict[str, int] = Field(..., description="Total refueling count")
    rental_distance: Dict[str, float] = Field(..., description="Total rental distance")
    rental_time: Dict[str, float] = Field(..., description="Total rental time")
    relocation_distance: Dict[str, float] = Field(..., description="Total relocation distance")
    refueling_distance: Dict[str, float] = Field(..., description="Total refueling distance")
    rental_revenue: Dict[str, float] = Field(..., description="Total rental revenue")
    profit: Dict[str, float] = Field(..., description="Total profit")
    score: float = Field(..., description="Evaluation score")
    subzone_cell_ids: list[str]
    @classmethod
    def create(cls, experiment_id: str, simulate_in: ParallelSimulateIn, kpis: Kpis):
        relocation_actions_proportion = (
            kpis.total_relocations_number / kpis.total_rent_number
            if kpis.total_rent_number > 0
            else 0
        )

        return cls(
            experiment_id=experiment_id,
            subzone_id=simulate_in.simulate_in.subzone.id,
            vehicle_count=len(simulate_in.simulate_in.vehicles),
            action_count=simulate_in.simulate_in.service_actions_nbr,
            relocation_actions_proportion=relocation_actions_proportion,
            rent_count={"total": kpis.total_rent_number},
            relocation_count={"total": kpis.total_relocations_number},
            refueling_count={"total": kpis.total_refueling_number},
            rental_distance={"total": kpis.total_rent_distance},
            rental_time={"total": kpis.total_rent_time},
            relocation_distance={"total": kpis.total_relocations_distance},
            refueling_distance={"total": kpis.total_refueling_distance},
            rental_revenue={"total": kpis.total_rent_revenue},
            profit={"total": kpis.total_profit},
            score=kpis.evaluate_result(),
            subzone_cell_ids=simulate_in.simulate_in.subzone.cell_ids,
        )
