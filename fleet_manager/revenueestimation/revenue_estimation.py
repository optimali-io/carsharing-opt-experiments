import logging
from typing import List

import numpy as np
import pandas as pd

from fleet_manager.fastsimulator.fast_simulator import FastSimulator

log = logging.getLogger(__name__)


class RevenueEstimation:
    """
    Contains hyperbolic function parameters array and revenue array.
    """

    def __init__(self):
        self.parameters_array: np.ndarray = None
        self.revenue: np.ndarray = None


def generate_revenue_estimation(fast_simulator: FastSimulator) -> RevenueEstimation:
    """
    Generates RevenueEstimation base on the FastSimulator results.
    """
    coeff_array: np.matrix = np.matrix([[1] * 10, [1 / (i + 1) for i in range(10)]])
    simulation_points: List[int] = [1, 2, 3]

    revenue_estimation = RevenueEstimation()
    revenue_estimation.parameters_array = _prepare_parameters_array(fast_simulator, simulation_points)
    revenue_estimation.revenue = _parameters_array_to_revenue(revenue_estimation.parameters_array, coeff_array)
    return revenue_estimation


def _prepare_parameters_array(
    fast_simulator: FastSimulator, simulation_points: List[int], function_parameters_nbr: int = 2
) -> np.ndarray:
    """
    Prepares hyperbolic function parameters base on FastSimulator results.
    """
    log.info("Starting to prepare parameters array")
    simulation_results = _run_simulations(fast_simulator, simulation_points)
    parameters_array = np.zeros((fast_simulator.cells_nbr(), function_parameters_nbr))
    for idx, row in simulation_results.iterrows():
        revenues = row[[str(point) for point in simulation_points]].values.tolist()
        parameters_array[int(row.cell)] = _get_hiperbolic_function_parameters(simulation_points, revenues)
    log.info("Prepared parameters array")
    return parameters_array


def _run_simulations(fast_simulator: FastSimulator, simulation_points: List[int]) -> pd.DataFrame:
    """
    Runs simulations, each simulation has all cars in single cell, number of cars is given in simulation_points.
    """
    cells = [i for i in range(fast_simulator.cells_nbr())]
    simulation_results = pd.DataFrame({"cell": cells})
    log.info("Starting to run simulations")
    for cars_nbr in simulation_points:
        location_rv = [[i] * cars_nbr for i in range(fast_simulator.cells_nbr())]
        location_rv = np.array(location_rv).astype(np.uint16)
        result = fast_simulator.simulate(location_rv)
        simulation_results[str(cars_nbr)] = result.revenue_row_simulation.mean(axis=1)
    log.info("Ended simulations")
    return simulation_results


def _get_hiperbolic_function_parameters(x: List[int], y: List[float]):
    """
    Fit hyperbolic function parameters to list of points.
    """
    tmp_A = []
    for i in range(len(x)):
        tmp_A.append([1, 1 / x[i]])
    A = np.matrix(tmp_A)
    r = np.matrix(y).T
    fit = (A.T * A).I * A.T * r
    # errors = r - A * fit
    parameters = [float(fit[0]), float(fit[1])]
    return parameters


def _parameters_array_to_revenue(parameters_array: np.ndarray, coeff_array: np.matrix) -> np.ndarray:
    """
    Calculate revenue in each cell for each number of cars.
    """
    revenue_matrix = np.matrix(parameters_array) * coeff_array
    return np.squeeze(np.asarray(revenue_matrix))
