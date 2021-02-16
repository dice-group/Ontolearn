import logging

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
TRACE = 5
logging.addLevelName(TRACE, 'TRACE')
