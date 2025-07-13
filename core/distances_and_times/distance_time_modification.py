import logging
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

log = logging.getLogger("core")


def calculate_time_traffic_factor(rents: pd.DataFrame, start_night_hour: int, end_night_hour: int) -> np.array:
    """
    Create 7x24 numpy array with time traffic factors.
    :param rents: pandas dataframe created from rents list from backend
    :param start_night_hour: hour when night starts (ex. 22)
    :param end_night_hour: hour_when_night_ends (ex. 4)
    :return 7x24 numpy array with time traffic factors
    """
    log.info("grouping rents by weekday and hour")
    group_columns = ["weekday", "start_hour", "distance", "driving_time_minutes"]
    grouped = rents[group_columns].groupby(["weekday", "start_hour"]).sum().reset_index()

    log.info("calculating average speed")
    grouped["avg_speed"] = 0.06 * grouped["distance"] / grouped["driving_time_minutes"]
    log.info(f"calculating average night speed for hours {start_night_hour} - {end_night_hour}")
    avg_night_spd = grouped.loc[(grouped.start_hour < end_night_hour) | (grouped.start_hour > start_night_hour)][
        "avg_speed"
    ].mean()
    grouped.loc[
        (grouped["start_hour"] >= start_night_hour) | (grouped["start_hour"] <= end_night_hour), "avg_speed"
    ] = avg_night_spd
    grouped.loc[(grouped["avg_speed"] > avg_night_spd), "avg_speed"] = avg_night_spd

    log.info("calculating time traffic factor")
    grouped["time_traffic_factor"] = avg_night_spd / grouped["avg_speed"]
    grouped.drop(["driving_time_minutes", "distance"], axis=1, inplace=True)

    grouped = grouped.round(3)
    log.info("converting time traffic factor to matrix")
    time_traffic_factor_array: np.array = np.ones((7, 24))

    for row in grouped.itertuples():
        time_traffic_factor_array[row.weekday, row.start_hour] = row.time_traffic_factor

    for i in range(7):
        log.info(f"factors for weekday {i}: {time_traffic_factor_array[i]}")

    assert np.sum(time_traffic_factor_array < 1) == 0

    return time_traffic_factor_array.astype(np.float64)


def fit_function(x: Union[np.array, float], a: float, b: float, c: float, d: float, e: float) -> Union[float, np.array]:
    """
    Function approximating rents distance as a function of a distance between cells.
    :param x: data point/s, numpy array or float
    :param a: function parameter
    :param b: function parameter
    :param c: function parameter
    :param d: function parameter
    :param e: function parameter
    """
    return (a * x * x + b * x + c) / (d * x + e)


def get_route_function_approximation(
    rents: pd.DataFrame, distance_matrix: np.array, bin_size: int, approx_func: callable = fit_function
):
    """
    Calculate parameters of distance approximating function.
    :param rents: pandas DataFrame with rents records. "start_cell_id" and "end_cell_id" must be already mapped to matrix index!
    :param distance_matrix: NxN numpy array with distances between cells.
    :param bin_size: bins for real distances, ex. 0-500, 500-1000
    :param approx_func: distance modifying function F such that F(road distance matrix) -> rent distance matrix.
    :return numpy array with parameters for given function
    """

    log.info("adding HERE distance to rents")
    rents["here_distance"] = rents.apply(lambda row: distance_matrix[row["start_cell_id"], row["end_cell_id"]], axis=1)

    max_here_dist = rents.here_distance.max()

    log.info("aggregating rents into bins by HERE distance")
    # aggregate rents to distance bins by distance between start and end cell
    rents.here_distance = pd.cut(x=rents.here_distance, right=False, bins=range(0, max_here_dist + bin_size, bin_size))
    rents.here_distance = rents.here_distance.apply(lambda i: i.left).astype(int)
    rents["n_rents"] = rents.here_distance
    rents["distance_std"] = rents["distance"]
    rents = (
        rents.groupby("here_distance")
        .agg({"distance": "mean", "n_rents": "count", "distance_std": "std"})
        .dropna()
        .reset_index()
    )

    log.info("calculating weights for rent data points")
    # calculate weights for data points
    sigma = (rents.distance_std.values + 1) / (rents.n_rents.values + 1)

    log.info("fitting function to data points")
    # fit
    popt, _ = curve_fit(approx_func, rents.here_distance, rents.distance, method="lm", sigma=sigma)

    return popt
