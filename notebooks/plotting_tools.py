import numpy as np
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

        #get the filenames for the endpoint perturbations
        endpoint_file_prefix = os.path.join(trajectory_directory, trajectory_prefix + "endpoint{endpoint_idx}.npy")
        self._endpoint_work_paths = [endpoint_file_prefix.format(endpoint_idx=lambda_state) for lambda_state in [0, 1]]

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
            traj_filename_dict = {int(filename.split(".")[-4]) : filename for filename in neq_traj_filenames[lambda_state]}
            self._neq_traj_filenames.append(traj_filename_dict)

    def get_nonequilibrium_trajectory(self, direction, trajectory_index):
        """
        Get a nonequilibrium trajectory corresponding to a particular direction (forward, 0->1, reverse 1->0),
        and run index.

        Parameters
        ----------
        direction : str
           "forward" or "reverse"
        trajectory_index : int
            Iteration number of protocol

        Returns
        -------
        nonequilibrium_trajectory : md.Trajectory
            the nonequilibrium trajectory
        """
        if direction=='forward':
            lambda_state = 0
        elif direction=='reverse':
            lambda_state = 1
        else:
            raise ValueError("direction must be either forward or reverse")

        nonequilibrium_trajectory_filename = self._neq_traj_filenames[lambda_state][trajectory_index]

        return md.load(nonequilibrium_trajectory_filename)

    @property
    def lambda_zero_traj(self):
        """
        Get the equilibrium trajectory corresponding to lambda=0

        Returns
        -------
        lambda_zero_traj : md.Trajectory object
            The equilibrium trajectory at lambda=0
        """
        lambda_zero_filename = self._trajectory_filename[0]
        return md.load(lambda_zero_filename)

    @property
    def lambda_one_traj(self):
        """
        Get the equilibrium trajectory corresponding to lambda=1

        Returns
        -------
        lambda_one_traj : md.Trajectory object
            The equilibrium trajectory at lambda=1
        """
        lambda_one_filename = self._trajectory_filename[1]
        return md.load(lambda_one_filename)

    @property
    def cumulative_work(self):
        """
        Get the cumulative work array for the nonequilibrium switching.
        The arrays have the indexing [lambda_state, protocol_index, step_index].

        Returns
        -------
        cumulative_works : np.array [2, n_protocols, n_steps_per_protocol]
            The cumulative work for the protocols that were run.
        """
        return self._cumulative_works