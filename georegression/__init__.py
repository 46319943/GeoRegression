import logging

from slab_utils.quick_logger import logger

# Change the format for file output
fileHandler = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)][0]
formatter4File = logging.Formatter(
    '[%(levelname)s] - %(asctime)s:\n%(message)s',
    '%Y-%m-%d %H:%M:%S')
fileHandler.setFormatter(formatter4File)
