import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import ontolearn
import logging
import logging.config

from ontolearn.utils import oplogging

logger = logging.getLogger(__name__)


def _try_load(fn):
    logging.config.fileConfig(fn, disable_existing_loggers=False)
    logger.debug("Loaded log config: %s", fn)


def _log_file(fn):
    file_name = datetime.now().strftime(fn)
    path_dirname = os.path.realpath(os.path.dirname(file_name))
    os.makedirs(path_dirname)
    log_dirs.append(path_dirname)
    return file_name


log_dirs = []


def setup_logging(config_file="logging.conf"):
    logging.x = SimpleNamespace(log_file=_log_file)
    logging.TRACE = oplogging.TRACE

    try:
        _try_load(Path(config_file).resolve())
    except KeyError:
        try:
            _try_load(Path(ontolearn.__path__[0], "..", config_file).resolve())
        except KeyError:
            print("Warning: could not find %s" % config_file, file=sys.stderr)
