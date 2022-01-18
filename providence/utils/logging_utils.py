# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
import sys

LOGFILE = "./outputs/log.log"
Path(LOGFILE).parent.mkdir(parents=True, exist_ok=True)

ch = logging.StreamHandler(sys.stdout)
fh = logging.handlers.RotatingFileHandler(LOGFILE, maxBytes=(1048576 * 5), backupCount=3)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] PROVIDENCE:%(levelname)s - %(message)s", handlers=[ch, fh])

logging.captureWarnings(True)
logger = logging.getLogger(__name__)
