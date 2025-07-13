import numpy as np


def predict_7_days_of_demand(demand_history_dhc: np.ndarray) -> np.ndarray:
    """
    Predict 7 days of demand. The mean prediction method is used.

    :param demand_history_dhc: demand history array
    :type demand_history_dhc: numpy.ndarray, shape=(D, H, C), dtype=numpay.float32

    :return: 7 days of demand prediction
    :rtype: numpy.ndarray, shape=(7, H, C), dtype=numpay.float32
    """

    days, hours, cells = demand_history_dhc.shape
    assert days % 7 == 0
    assert days // 7 > 0
    assert hours == 24
    assert demand_history_dhc.dtype == np.float32

    demand_history_wdhc = demand_history_dhc.reshape((-1, 7, hours, cells))
    demand_prediction_dhc = np.mean(demand_history_wdhc, axis=0)
    return demand_prediction_dhc
