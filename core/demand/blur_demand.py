import logging

import numpy as np

log = logging.getLogger("core")


def blur_demand_history(
    demand_history_dhc: np.ndarray, distance_cc: np.ndarray, blur_distance: float, blur_factor: float
) -> np.ndarray:
    """
    Blur demand in distance dimension. Add <blur_factor> * <demand from cell C> to every cell within
    distance of <blur_distance> from cell C.
    :param demand_history_dhc: demand history array (np.ndarray) of shape n_days x 24 hours x n_cells
    :param distance_cc: np.ndarray with distances in meters between cells of shape n_cells x n_cells
    :param blur_distance: cells within this distance from cell C will have added demand
    :param blur_factor: factor that demand from cell C will be multiplied before being added to any cell.
    """
    days_nbr, hours_nbr, cells_nbr = demand_history_dhc.shape
    assert demand_history_dhc.dtype == np.float64

    assert distance_cc.shape == (cells_nbr, cells_nbr)
    assert distance_cc.dtype == np.uint32

    demand_history_blurred_dhc = demand_history_dhc.astype(np.float32)
    for day in range(days_nbr):
        for hour in range(hours_nbr):
            in_demand = demand_history_dhc[day, hour]
            out_demand = np.copy(in_demand)
            for cell_id, cell_value in enumerate(in_demand):
                if cell_value >= 1:
                    close_mask = np.logical_and(distance_cc[cell_id] > 10, distance_cc[cell_id] < blur_distance)
                    close_indexes = np.where(close_mask)
                    out_demand[close_indexes] += cell_value * blur_factor
            demand_history_blurred_dhc[day, hour, :] = out_demand
    log.info(f"blurred demand for {days_nbr} days")

    return demand_history_blurred_dhc
