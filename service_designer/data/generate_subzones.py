import json
import logging
from typing import Dict, List

import pandas as pd

from core.db.data_access_facade_basic import DataAccessFacadeBasic
from core.utils.id_to_index_mapper import IdToIndexMapper
from core.utils.nptools import rearrange_from_day_hour_cell_to_hour_cell
from service_designer.data.data_config import ExperimentConfig, DataConfig
from service_designer.data.subzones import Subzone
from service_designer.tools import cells_order_by_demand

logger = logging.getLogger("service_designer")


class SubzonesGenerator:
    """
    Subzones generator creates Subzone objects ...

    :param config: dictionary of generator configuration:
                    - id - SubzonesGeneratorConfig id
                    - data_config - dictionary of data configuration
                    - subzone_count - number of subzones to be created
                    - minimum_cells_in_subzone - minimum number of cells in subzone
                    - subzone_cells_origin - way of cells sorting order
    """

    def __init__(self, experiment_config: ExperimentConfig):
        self.id = experiment_config.name
        self.data_config: DataConfig = experiment_config.data_config
        self.subzone_count: int = experiment_config.subzones_generator_config.subzone_count
        self.minimum_cells_in_subzone: int = experiment_config.subzones_generator_config.minimum_cells_in_subzone
        self.cells_sort_by: str = experiment_config.subzones_generator_config.subzone_cells_origin
        self.cell_ids = self.data_config.base_zone.cell_ids
        self.id_mapper = IdToIndexMapper(self.cell_ids)
        self.cells_order: pd.DataFrame = None
        self.experiment_config: ExperimentConfig = experiment_config

    def prepare_cells_order(self):
        """
        Function prepares DataFrame of sorted cells IDs -
        in order depending on chosen carsharing indicator (demand, profit or number of rents)
        """

        if self.cells_sort_by == "by_demand":
            # cells by typical demand
            daf = DataAccessFacadeBasic()

            rent_demand = daf.find_demand_prediction_array(self.experiment_config.get_experiment_zone_id(), self.data_config.end_date)
            rent_demand = rearrange_from_day_hour_cell_to_hour_cell(
                rent_demand, days=7, hours=24, cells=len(self.cell_ids)
            )

            act_zone_mask = self.id_mapper.get_zone_mask(subzone_cell_ids=self.cell_ids)
            self.cells_order = cells_order_by_demand(rent_demand, act_zone_mask)

        logger.info(f"Cells ordered {self.cells_sort_by}")

    def generate_subzones(self):
        """
        Generate Subzones basing on cells quality. The greatest subzone includes oll cells from given BaseZone.
        Each next Subzone is smaller, the worst cells are
        """
        logger.info(f"Generate Subzones basing on SubzonesGenerator: {self.id}")
        self.prepare_cells_order()
        max_cell_count = len(self.cells_order)
        cell_count_difference = (max_cell_count - self.minimum_cells_in_subzone) // self.subzone_count + 1

        indices: List[int] = list(self.cells_order)
        cell_ids: List[str] = self.id_mapper.revert_indices_to_cell_ids(indices)
        subzones = []
        for i in range(self.subzone_count):
            subzone_cells = cell_ids[: self.minimum_cells_in_subzone + i * cell_count_difference]
            subzone: Subzone = Subzone(
                id=f"Subzone {self.id}.{i}.{len(subzone_cells)}{self.cells_sort_by}",
                cell_ids=subzone_cells,
            )

            logger.info(f"Saving Subzone: {subzone.id}")

            subzones.append(subzone)
        return subzones
