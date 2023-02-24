"""
Utility classes and functions for logging in perses
"""

import logging
import yaml


class YAMLFormatter(logging.Formatter):
    """Custom formatter to output logs in YAML format"""

    def format(self, record):
        """Format to dump into the yaml file.

        Adds more information if DEBUG is the level of the record.
        """
        if record.levelname == "DEBUG":
            data = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
        else:
            data = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "message": record.getMessage(),
            }
        return yaml.dump(data)
