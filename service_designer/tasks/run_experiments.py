import logging

from config import huey, get_task_logging_adapter
from core.db.data_access_facade_basic import DataAccessFacadeBasic
from service_designer.data.data_config import ExperimentConfig
from service_designer.data.get_data import prepare_base_data_if_not_exist
from service_designer.experiments.experiments import Experiment

base_logger = logging.getLogger("service_designer")


@huey.task(context=True)
def run_gridsearch(experiment_id: str, prepare_data: bool = True, task=None):
    """Prepare data if necessary and run experiment for provided experiment id"""
    log = get_task_logging_adapter(base_logger, task)
    daf = DataAccessFacadeBasic()
    experiment_config: ExperimentConfig = daf.get_experiment_config_by_id(experiment_id)

    log.info(f"downloading data config for experiment {experiment_id}")
    if prepare_data:
        prepare_base_data_if_not_exist(experiment_config)

    log.info("creating experiment object")
    exp = Experiment(
        experiment_config=experiment_config
    )
    log.info("run experiment")
    exp.run_experiment()
    return True
