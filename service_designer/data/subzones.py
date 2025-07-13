from typing import Dict, List

from core.utils.id_to_index_mapper import IdToIndexMapper


class Subzone:
    """
    Subzone model.

    :param config: subzone config dictionary
    """

    def __init__(self, id: str, cell_ids: List[str]):
        self.id: str = id
        self.cell_ids: List[str] = cell_ids
        self.zone_mask = None

    def generate_zone_mask(self, base_zone_cell_ids) -> None:
        """Create zone mask for given BaseZone based on Subzone cell ids"""
        id_mapper = IdToIndexMapper(base_zone_cell_ids)
        self.zone_mask = id_mapper.get_zone_mask(subzone_cell_ids=self.cell_ids)
