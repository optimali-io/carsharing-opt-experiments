import numpy as np


def cells_order_by_demand(rent_demand: np.array, act_zone_mask: np.array) -> np.array:
    """
    Function returning order of subzone cells determinated by demand.

    :param rent_demand: array of demand per cell
    :param act_zone_mask: subzone mask of cells
    """
    quality_c = np.sum(rent_demand, axis=0)
    order_i = np.flip(np.argsort(quality_c), axis=0)
    return np.array([i for i in order_i if act_zone_mask[i]], dtype=np.uint32)
