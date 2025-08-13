import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


def get_env_or_raise(var):
    """Gen an environmental variable or raise exception if it can't be found."""
    res = os.getenv(var)
    if res is None:
        raise OSError("Env variable not configured")
    return res


class Settings(BaseSettings):
    """Class that aggregates apps parameters."""

    MAIN_PATH: str = get_env_or_raise("MAIN_PATH")
    APP_NAME: str = "science_core"
    REDIS_URL: str = get_env_or_raise("REDIS_URL")
    DATA_DIR: str = MAIN_PATH + "/data"
    RESULTS_DIR: str = MAIN_PATH + "/results"
    ZONES_PATH: str = DATA_DIR + "/zones"
    SPATIAL_INDEX_PATH: str =DATA_DIR + "/spatial_index/spatial_index"
    REFERENCE_GRID_PATH: str = DATA_DIR + "/reference_grid"


    SDESIGNER_DIR: str = MAIN_PATH + "/sdesigner"
    SDESIGNER_RESULTS_DIR: str = SDESIGNER_DIR + "/experiments_results"
    SDESIGNER_EXP_ZONES_DIR_NAME: str = "sdesigner_dataconfig_zones"

    # HERE
    HERE_API_KEY: str = get_env_or_raise("HERE_API_KEY")
    HERE_APP_ID: str = get_env_or_raise("HERE_APP_ID")

    # configuration
    demand_lag_days: int = 7
    directions_lag_weeks: int = 1

