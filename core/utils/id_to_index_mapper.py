from typing import List, Union

import numpy as np


class IdToIndexMapper:
    """
    Class responsible for mapping ids to their indices in ascending order.
    """

    def __init__(self, original_ids: List[Union[str, int]], sort=True):
        if sort:
            self._original_ids = sorted(original_ids)
        else:
            self._original_ids = list(original_ids)

    def map_ids_to_indices(self, input_cell_ids: List[str]) -> List[int]:
        """
        Map ids to integer values from (0, ..., n) so they can be used as matrix indices.
        Function should be used after loading events data (Vehicles, Rents, AppStarts) and before running
        data_preparation like calculating demand, directions, revenue, creating subzones
        or setting up fleet for genetic algorithm.

        If cell is not in original ids, it will be mapped to -1. Remember to remove it
        """

        indices = [
            self._original_ids.index(cell_id) if cell_id in self._original_ids else -1 for cell_id in input_cell_ids
        ]
        return indices

    def map_id_to_index(self, input_cell_id: str) -> int:
        """
        Map id to integer value from (0, ..., n) so it can be used as matrix index.
        Function should be used after loading events data (Vehicles, Rents, AppStarts) and before running data preparation
        data_preparation like calculating demand, directions, revenue or setting up fleet for genetic algorithm.
        """
        if input_cell_id not in self._original_ids:
            raise ValueError(f'Cell id: "{input_cell_id}" not in base ids.')
        return self._original_ids.index(input_cell_id)

    def revert_indices_to_cell_ids(self, indices: List[int]) -> List[str]:
        """
        Map list of matrix indices to corresponding cell ids.
        :param indices: list of matrix indices to map
        :return: list of corresponding cell ids
        """
        return [self._original_ids[idx] for idx in indices]

    def revert_index_to_cell_id(self, index: int) -> str:
        """
        Map index to cell id.
        :param index: index
        :return: cell id
        """
        return self._original_ids[index]

    def get_zone_mask(self, subzone_cell_ids: List[str]) -> np.ndarray:
        """
        Create a boolean vector of size N, where N is a number of cells in base zone, which values defines
        if given cell is contained by a subzone. (True if cell in subzone, False otherwise).
        :param subzone_cell_ids: a list containing string representations of subzone cell ids (ex. 19.3682-51.7983).
        This list doesn't have to be sorted.
        """
        subzone_indices: List[int] = self.map_ids_to_indices(sorted(subzone_cell_ids))
        mask = np.zeros(len(self._original_ids), dtype=bool)
        mask[subzone_indices] = True
        return mask
