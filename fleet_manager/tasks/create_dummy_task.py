import logging

from app.config import backend, huey
from app.log import get_task_logging_adapter
from fleet_manager.external.backend.model import CreateTaskRequest

base_logger = logging.getLogger("fleet_manager")


@huey.task(context=True)
def create_dummy_task(task=None):
    """Sample task(huey task) that reads data from Backend and creates dummy Task(service team task) in Backend"""
    log = get_task_logging_adapter(base_logger, task)
    log.info("Start")

    log.info("Create Task")
    content = {}
    create_task_req = CreateTaskRequest(
        title="Dummy Task Title",
        description="Dummmy Task Description",
        vehicle=1,
        team=1,
        content=content,
    )
    create_task_resp = backend.create_task(create_task_req)
    log.info(create_task_resp)

    log.info("Stop")
