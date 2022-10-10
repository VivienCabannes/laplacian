"""
Configuration file.

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2022, Vivien Cabannes
"""
import logging
from pathlib import Path

LOGS_PATH = Path('/') / "checkpoint" / "vivc" / "logs" / "ssl"
SAVE_PATH = Path.home() / "code" / "ssl" / "savings"


logging_level = logging.DEBUG
logging_config = {
    "format": "{asctime} {levelname:.1s} [{name}: {lineno}] {message}",
    "style": "{",
    "datefmt": "%H:%H:%S",
}
