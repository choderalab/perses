#!/usr/bin/env python

import logging
import os
import random
import time
import urllib.request

from http.client import RemoteDisconnected

# Setting logging level config
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
# logging.basicConfig(
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     level=LOGLEVEL,
#     datefmt='%Y-%m-%d %H:%M:%S')
_logger = logging.getLogger()
_logger.setLevel(LOGLEVEL)


# TODO: retrieve and fetch should be merged in a single function avoiding redundant code
def retrieve_file_url(url, retries=5):
    """Retrieves a file from url. Retries if it encounters an undesired exception. """
    remaining_tries = retries

    while remaining_tries > 0:
        try:
            file_path, _ = urllib.request.urlretrieve(url)
        except (urllib.error.URLError, urllib.error.HTTPError, RemoteDisconnected) as e:
            wait_time = random.uniform(1, 3)
            _logger.warning(f"Error downloading {url}. Retrying in {wait_time} seconds...")
            remaining_tries = remaining_tries - 1
            last_error = e
            time.sleep(wait_time)  # wait for a random time between 1-3 seconds
            continue
        else:
            break
    else:
        _logger.error(f"Could not fetch data.")
        raise last_error
    return file_path


def fetch_url_contents(url, retries=5):
    """Fetch contents from url. Retries if it encounters an undesired exception. """
    remaining_tries = retries

    while remaining_tries > 0:
        try:
            file_contents = urllib.request.urlopen(url)
        except (urllib.error.URLError, urllib.error.HTTPError, RemoteDisconnected) as e:
            wait_time = random.uniform(1, 3)
            _logger.warning(f"Error downloading {url}. Retrying in {wait_time} seconds...")
            remaining_tries = remaining_tries - 1
            last_error = e
            time.sleep(wait_time)  # wait for a random time between 1-3 seconds
            continue
        else:
            break
    else:
        _logger.error(f"Could not fetch data.")
        raise last_error
    return file_contents
