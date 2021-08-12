#!/usr/bin/env python

# ======================================================================
# MODULE DOCSTRING
# ======================================================================

"""
Test that the examples in the repo run without errors.
"""

# ======================================================================
# GLOBAL IMPORTS
# ======================================================================

import os
import pathlib
import pytest
import subprocess
import tempfile


ROOT_DIR_PATH = pathlib.Path(__file__).joinpath("../../../").resolve()


def run_script_file(file_path, cmd_args=None):
    """Run through the shell a python script."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        cmd = ["python", file_path]
        # Extend cmd list with given cmd_args
        if cmd_args:
            cmd.extend(cmd_args)
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            raise Exception(f"Example {file_path} failed")


def find_example_scripts():
    """Find all Python scripts, excluding Jupyter notebooks, in the examples folder.
    Returns
    -------
    example_file_paths : List[str]
        List of full paths to python scripts to execute.
    """
    examples_dir_path = ROOT_DIR_PATH.joinpath("examples")

    example_file_paths = []
    for example_file_path in examples_dir_path.glob("*/*.py"):
        example_file_paths.append(example_file_path.as_posix())

    return example_file_paths


# ======================================================================
# TESTS
# ======================================================================
@pytest.mark.parametrize("example_file_path", find_example_scripts())
def test_examples(example_file_path):
    """Test that the example run without errors."""
    run_script_file(example_file_path)
