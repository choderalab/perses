import numpy as np
import nglview
from bokeh.plotting import figure, output_file, show
import simtk.unit as unit
import mdtraj as md
import os
import glob

class NonequilibriumSwitchingAnalysis(object):
    """
    This is a helper class for analyzing nonequilibrium switching data
    """

    def __init__(self, trajectory_directory, trajectory_prefix):
        """

        Create a NonequilibriumSwitchingAnalysis class

        Parameters
        ----------
        trajectory_directory : str
            The location of the trajectories specified in the yaml setup file
        trajectory_prefix : str
            The prefix for the files written by this calculation
        """
        self._trajectory_directory = trajectory_directory
        self._trajectory_prefix = trajectory_prefix

        #get the names of the files
        self._trajectory_filename = {lambda_state: os.path.join(self._trajectory_directory, trajectory_prefix+"lambda%d" % lambda_state + ".h5") for lambda_state in [0,1]}
        self._neq_traj_filename_pattern = {lambda_state: os.path.join(self._trajectory_directory, trajectory_prefix + ".*.neq.lambda%d" % lambda_state + ".h5") for lambda_state in [0,1]}
        self._neq_work_filename_pattern = {lambda_state: os.path.join(self._trajectory_directory, trajectory_prefix + ".*.neq.lambda%d" % lambda_state + ".cw.npy") for lambda_state in [0,1]}

        #generate filenames from the patterns
        cum_work_filenames = [glob.glob(self._neq_work_filename_pattern[lambda_state]) for lambda_state in [0,1]]
        neq_traj_filenames = [glob.glob(self._neq_traj_filename_pattern[lambda_state]) for lambda_state in [0,1]]

        #sort the cumulative work files by iteration number and load them:
        stacked_work_arrays = []
        for lambda_state in [0,1]:
            cumulative_work_filenames = cum_work_filenames[lambda_state]

            #sort by the iteration number in the filename. The lambda function takes the 5th from the end element of the filename after
            #splitting by the period, which is the iteration index
            cumulative_work_filenames_sorted = sorted(cumulative_work_filenames, key=lambda filename: int(filename.split(".")[-5]))

            #load these with numpy
            sorted_work_arrays = [np.load(work_filename) for work_filename in cumulative_work_filenames_sorted]

            #stack the work arrays
            stacked_work_arrays.append(np.stack(sorted_work_arrays))

        #cumulative_works is now [lambda_index, iteration_number, step_count]
        self._cumulative_works = np.stack(stacked_work_arrays)

        #Now prepare the filenames for the trajectories. Don't load them all because it might be quite large.
        self._neq_traj_filenames = []

        for lambda_state in [0,1]:
            #index them by their iteration number
            traj_filename_dict = {int(filename.split(".")[-4]) for filename in neq_traj_filenames[lambda_state]}
            self._neq_traj_filenames.append(traj_filename_dict)

