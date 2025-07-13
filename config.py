import logging

from huey import RedisHuey
from huey.api import Task
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
)

# load settings
settings = Settings()

# set database
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
Base = declarative_base()

# set huey
huey = RedisHuey(settings.APP_NAME, url=settings.REDIS_URL)
solo_huey = RedisHuey(f"{settings.APP_NAME}_solo", url=settings.REDIS_URL)



def get_task_logging_adapter(logger: logging.Logger, task: Task | None=None):
    """Returns tasks logger with task info"""
    if not task:
        return logger
    adapter = logging.LoggerAdapter(logger, {"task_id": task.id, "task_name": task.name})
    return adapter