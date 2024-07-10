# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""Logger configuration."""
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


def _log_file(fn):  # pragma: no cover
    file_name = datetime.now().strftime(fn)
    path_dirname = os.path.realpath(os.path.dirname(file_name))
    os.makedirs(path_dirname)
    log_dirs.append(path_dirname)
    return file_name


log_dirs = []


def setup_logging(config_file="ontolearn/logging.conf"):
    """Setup logging.

    Args:
        config_file (str): Filepath for logs.
    """
    logging.x = SimpleNamespace(log_file=_log_file)
    logging.TRACE = oplogging.TRACE

    try:
        _try_load(Path(config_file).resolve())
    except KeyError:  # pragma: no cover
        try:
            _try_load(Path(ontolearn.__path__[0], "..", config_file).resolve())
        except KeyError:
            print("Warning: could not find %s" % config_file, file=sys.stderr)
