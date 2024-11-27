import logging

from . import constants
from . import models
from . import routers
from . import schemas
from . import services
from . import static
from . import templates
from . import utils
from . import config
from . import database
from . import main

logging.basicConfig(level=logging.INFO, format="CONSOLE: %(message)s")

__all__ = [
    "constants",
    "models",
    "routers",
    "schemas",
    "services",
    "static",
    "templates",
    "utils",
    "config",
    "database",
    "main",
]
