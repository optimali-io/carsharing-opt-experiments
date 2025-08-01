import datetime as dt
import logging
from datetime import timedelta

from config import huey
from core.db.data_access_facade_basic import DataAccessFacadeBasic
from core.tasks.data_preparation import calculate_demand_prediction, calculate_historical_demand_for_date_range, \
    calculate_directions, calculate_revenue_matrix
from core.tasks.distance_time_modification import run_distance_modification_approximation, run_traffic_factors_task
from fleet_manager.data_model.config_models import ScienceConfig
from fleet_manager.tasks.optimisation import load_zone_data, run_genetic_algorithm
from service_designer.tasks.run_experiments import run_gridsearch
