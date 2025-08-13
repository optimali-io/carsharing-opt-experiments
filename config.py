import logging

from huey import RedisHuey
from huey.api import Task

from settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Set the logging level
)

# load settings
settings = Settings()

# set huey
huey = RedisHuey(settings.APP_NAME, url=settings.REDIS_URL)
solo_huey = RedisHuey(f"{settings.APP_NAME}_solo", url=settings.REDIS_URL)



def get_task_logging_adapter(logger: logging.Logger, task: Task | None=None):
    """Returns tasks logger with task info"""
    if not task:
        return logger
    adapter = logging.LoggerAdapter(logger, {"task_id": task.id, "task_name": task.name})
    return adapter