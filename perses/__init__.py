#
import os
def get_datadir():
    """Returns the data directory of this package"""
    return os.path.join(os.path.dirname(__file__), 'data')
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
