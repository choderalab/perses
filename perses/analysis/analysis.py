"""
Analysis tools for perses automated molecular design.

TODO
----
* Analyze all but last iteration to ensure we can analyze a running simulation?

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

import os, os.path
import sys, math
import numpy as np
import copy
import time
import netCDF4 as netcdf
from openeye import oeiupac, oechem
import pickle
import json
import itertools
import pymbar
from perses import storage

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# ANALYSIS
################################################################################

class Analysis(object):
    """Analysis tools for perses automated design.

    """
    def __init__(self, storage_filename):
        """Open a storage file for analysis.

        """
        # TODO: Replace this with calls to storage API
        self._storage = storage.NetCDFStorage(storage_filename, mode='r')
        self._ncfile = self._storage._ncfile
        self.storage_filename = storage_filename
        self._environments = self.get_environments()
        self._n_exen_iterations = {}
        for environment in self._environments:
            self._n_exen_iterations[environment] = len(self._ncfile.groups[environment]['ExpandedEnsembleSampler']['logP_accept'])
        self._state_transitions, self._visited_states = self._get_state_transitions()
        self._logP_accepts = {}


    def get_environments(self):
        """Return a list of environments in storage file.

        Returns
        -------
        environments : list of str
           List of environment names in storage (e.g. []'explicit-complex', 'explicit-ligand'])

        """
        environments = list()
        for group in self._ncfile.groups:
            environments.append(str(group))
        return environments

    def _state_transition_to_iupac(self, state_transition):
        """
        Convenience function to convert SMILES to IUPAC names

        Parameters
        ----------
        state_transition : (str, str)
            Pair of smiles strings for the state transition

        Returns
        -------
        state_transition_iupac : [str, str]
            The pair of molecules in IUPAC names
        """
        state_transition_iupac = []
        for state in state_transition:
            mol = oechem.OEMol()
            oechem.OESmilesToMol(mol, state)
            iupac = oeiupac.OECreateIUPACName(mol)
            state_transition_iupac.append(iupac)

        return state_transition_iupac

    def plot_work_trajectories(self, environment, filename):
        """
        Plot the NCMC work trajectories for the given environment and each attempted transition

        Parameters
        ----------
        environment : str
            Name of environment
        filename : str
            Name of output file
        """
        w_t = {state_transition : [] for state_transition in self._state_transitions[environment]}

        for iteration in range(self._n_exen_iterations[environment]):
            logP_ncmc_trajectory = self._ncfile.groups[environment]['NCMCEngine']['protocolwork'][iteration, :]
            state_key = self._storage.get_object(environment, "ExpandedEnsembleSampler", "state_key", iteration)
            proposed_state_key = self._storage.get_object(environment, "ExpandedEnsembleSampler", "proposed_state_key", iteration)
            if state_key == proposed_state_key:
                continue
            w_t[(state_key, proposed_state_key)].append(-logP_ncmc_trajectory)

        w_t_stacked = {state_transition: np.stack(work_trajectories) for state_transition, work_trajectories in w_t.items()}

        with PdfPages(filename) as pdf:
            sns.set(font_scale=2)
            for state_transition, work_array in w_t_stacked.items():

                fig = plt.figure(figsize=(28, 12))
                ax1 = sns.tsplot(work_array, color="Blue")

                iupac_transition = self._state_transition_to_iupac(state_transition)

                plt.title("{} => {} transition {} work trajectory".format(iupac_transition[0], iupac_transition[1], "NCMC"))
                plt.xlabel("step (1fs)")
                plt.ylabel("Work / kT")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

    def plot_sams_weights(self, environment):
        """
        Plot the trajectory of SAMS weights
        :param environment:
        :return:
        """
        pass

    def plot_chemical_trajectory(self, environment, filename):
        """
        Plot the trajectory through chemical space.

        Parameters
        ----------
        environment : str
            the name of the environment for which the chemical space trajectory is desired
        """
        chemical_state_trajectory = self.extract_state_trajectory(environment)

        visited_states = list(set(chemical_state_trajectory))

        state_trajectory = np.zeros(len(chemical_state_trajectory))
        for idx, chemical_state in enumerate(chemical_state_trajectory):
            state_trajectory[idx] = visited_states.index(chemical_state)

        with PdfPages(filename) as pdf:
            sns.set(font_scale=2)
            fig = plt.figure(figsize=(28, 12))
            plt.subplot2grid((1,2), (0,0))
            ax = sns.scatterplot(np.arange(len(state_trajectory)), state_trajectory)
            plt.yticks(np.arange(len(visited_states)), visited_states)

            plt.title("Trajectory through chemical space in {}".format(environment))
            plt.xlabel("iteration")
            plt.ylabel("chemical state")
            plt.tight_layout()

            plt.subplot2grid((1,2), (0,1))
            ax = sns.countplot(y=state_trajectory)

            pdf.savefig(fig)
            plt.close()

    def get_free_energies(self, environment):
        """
        Estimate the free energies between all pairs with bidirectional transitions of chemical states in the
        given environment

        Parameters
        ----------
        environment : str
            The name of the environment for which free energies are desired

        Returns
        -------
        free_energies : dict of (str, str): [float, float]
            Dictionary of pairwaise free energies and their uncertainty, computed with bootstrapping
        """
        logP_without_sams = self.extract_logP_values(environment, "logP_accept", subtract_sams=True)
        free_energies = {}
        n_bootstrap_iterations = 10000000

        for state_pair, logP_accepts in logP_without_sams.items():
            w_F = logP_accepts[0]
            w_R = -logP_accepts[1]
            bootstrapped_bar = np.zeros(n_bootstrap_iterations)
            for i in range(n_bootstrap_iterations):
                resampled_w_F = np.random.choice(w_F, len(w_F), replace=True)
                resampled_w_R = np.random.choice(w_R, len(w_R), replace=True)

                [df, ddf] = pymbar.BAR(resampled_w_F, resampled_w_R)
                bootstrapped_bar[i] = df

            free_energies[state_pair] = [np.mean(bootstrapped_bar), np.std(bootstrapped_bar)]

        return free_energies


    def _get_state_transitions(self):
        """
        Find the set of unique state transitions in each environment. This will be useful to retrieve various
        logP quantities.

        Returns
        -------
        state_transitions_dict : dict of str: set of (str, str) tuple
            The set of state transitions that were attempted in each environment. This counts (s1, s2) and (s2, s1) as separate.
        visited_states_dict : dict of str: set of str
            The set of states that were actually visited in each environment.
        """
        state_transitions_dict = {}
        visited_states_dict = {}
        for environment in self._environments:
            # first, find the set of unique state transitions:
            state_transition_list = []
            visited_states = []
            n_iterations = self._n_exen_iterations[environment]
            for iteration in range(n_iterations):
                state_key = self._storage.get_object(environment, "ExpandedEnsembleSampler", "state_key", iteration)
                proposed_state_key = self._storage.get_object(environment, "ExpandedEnsembleSampler",
                                                              "proposed_state_key", iteration)

                visited_states.append(state_key)
                # if they are the same (a self-proposal) just continue
                if state_key == proposed_state_key:
                    continue

                state_transition_list.append((state_key, proposed_state_key))

            # get the unique transitions:
            state_transition_set = set(state_transition_list)
            state_transitions_dict[environment] = state_transition_set
            visited_states_dict[environment] = set(visited_states)

        return state_transitions_dict, visited_states_dict

    def write_trajectory(self, environmnent, pdb_filename):
        """Write the trajectory of sampled configurations and chemical states.

        Returns
        -------
        environment : str
           Environment name to write trajectory for
        pdbfile : str
           Name of PDB file to generate.

        """
        # TODO
        pass

    def extract_logP_values(self, environment, logP_accept_component, subtract_sams=False):
        """
        Extract the requested logP_accept component from the ExpandedEnsembleSampler
        in the requested environment

        Parameters
        ----------
        environment : str
            The name of the environment
        logP_accept_component : str
            The name of the component of the acceptance probability that we want
        subtract_sams : bool, optional, default False
            Whether to subtract the SAMS weights corresponding to the same iteration. Useful for logP_accept.

        Returns
        -------
        logP_values : dict of (str, str) : list of float
             A dictionary for each state transition, with a list of the requested logP_accept component
        """
        n_iterations = self._n_exen_iterations[environment]

        logP_values = {state_transition: [] for state_transition in self._state_transitions[environment]}

        #loop through the iterations and
        for iteration in range(n_iterations):
            state_key = self._storage.get_object(environment, "ExpandedEnsembleSampler", "state_key", iteration)
            proposed_state_key = self._storage.get_object(environment, "ExpandedEnsembleSampler", "proposed_state_key", iteration)

            #if they are the same (a self-proposal) just continue
            if state_key == proposed_state_key:
                continue
            #retreive the work value (negative logP_work) and add it to the list of work values for that transition
            logP = self._ncfile.groups[environment]['ExpandedEnsembleSampler'][logP_accept_component][iteration]

            if subtract_sams:
                sams_weight = self._ncfile.groups[environment]['ExpandedEnsembleSampler']['logP_sams_weight'][iteration]
                logP = logP - sams_weight

            logP_values[(state_key, proposed_state_key)].append(logP)

        return logP_values

    def _prepare_logP_accept(self, environment):
        """
        Organize and retrieve the log acceptance probabilities for each of the transitions in the environment.

        Parameters
        ----------
        environment : str
            The name of the environment

        Returns
        -------
        logP_accept_dict : dict of (str, str) : list of 2 np.array
            A dictionary with a list of 2 np.arrays, one for s1->s2 logP_accept, another for s2->s1
            logP_accepts have had their SAMS weights subtracted if relevant
        """
        logP_accept_values = self.extract_logP_values(environment, "logP_accept", subtract_sams=True)

        logP_accept_dict = {}

        for state_pair in itertools.combinations(self._visited_states, 2):
            try:
                forward_logP = np.array(logP_accept_values[(state_pair[0], state_pair[1])])
                reverse_logP = np.array(logP_accept_values[(state_pair[1], state_pair[0])])
            except KeyError:
                continue
            logP_accept_dict[state_pair] = [forward_logP, reverse_logP]

        return logP_accept_dict


    def extract_state_trajectory(self, environment):
        """
        Extract the trajectory in chemical state space

        Parameters
        ----------
        environment : str
            The environment for which the chemical state is desired
        chemical_state_trajectory : list of str
            The trajectory in chemical space for the given environment

        Returns
        -------
        chemical_state_traj : list of str
            List of chemical states that were visited
         """
        n_iterations = self._n_exen_iterations[environment]
        chemical_state_traj = []
        for iteration in range(n_iterations):
            chemical_state = self._storage.get_object(environment, "ExpandedEnsembleSampler", "state_key", iteration)
            chemical_state_traj.append(chemical_state)

        return chemical_state_traj


    def plot_ncmc_work_distributions(self, environment, output_filename):
        """
        Plot the forward and reverse work distributions for NCMC switching in the given environment

        Parameters
        ----------
        environment : str
            The name of the environment for which NCMC work should be plotted
        output_filename : str
            The name of the PDF file to output
        """

        #get the unique transitions:
        state_transition_set = self._state_transitions[environment]
        visited_states_set = self._visited_states[environment]

        logP_values = self.extract_logP_values(environment, "logP_ncmc_work")

        #now loop through all the state pairs to plot each
        with PdfPages(output_filename) as pdf:
            sns.set(font_scale=2)
            for state_pair in itertools.combinations(visited_states_set, 2):

                iupac_pair = self._state_transition_to_iupac(state_pair)

                try:
                    #use the negative for the forward work because the logP contribution of the work is -work
                    forward_work = -np.array(logP_values[(state_pair[0], state_pair[1])])
                    reverse_work = np.array(logP_values[(state_pair[1], state_pair[0])])
                except KeyError:
                    continue

                fig = plt.figure(figsize=(28, 12))
                ax1 = sns.distplot(forward_work, kde=True, color="Blue")
                ax2 = sns.distplot(-reverse_work, color='Red', kde=True)
                plt.title("{} => {} transition {} work".format(iupac_pair[0], iupac_pair[1], "NCMC"))
                plt.xlabel("Work / kT")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()


    def plot_exen_logp_components(self, environment, filename_prefix=None, logP_range=20, nbins=20):
        """
        Generate histograms of each component of Expanded Ensemble log acceptance probability

        Arguments:
        ----------
        environment : str
            The environment to use
        filename_prefix : str, OPTIONAL, default = None
            if specified, each plot is saved as '{0}-{1}'.format(filename_prefix, component)
        logP__range : float, optional, default=None
            If specified, will set logP range to [-logP_range, +logP_range]
        nbins : int, optional, default=20
            Number of bins to use for histogram.
        Each histogram will be saved to {component name}.png
        TODO: include input filename
            storage ncfile has different hierarchy depending on which samplers are defined;
            this probably only works without SAMS sampling (otherwise top level groups are
            environments)

        """
        ee_sam = self._ncfile.groups[environment]['ExpandedEnsembleSampler']

        # Build a list of all logP components to plot:
        components = list()
        # Always show logP_accept
        components.append('logP_accept')
        # Summarize other logP groups
        for name in ee_sam.variables.keys():
            if name.startswith('logP_groups'):
                components.append(name)

        if filename_prefix is None:
            filename_prefix = self.storage_filename.split('.')[0]
        filename = '{0}-logP-components.pdf'.format(filename_prefix)
        with PdfPages(filename) as pdf:
            logps = dict()
            for component in components:
                try:
                    niterations = ee_sam.variables[component].shape[0]
                except:
                    continue
                logps[component] = np.zeros(niterations, np.float64)
                for n in range(niterations):
                    logps[component][n] = ee_sam.variables[component][n]
                # Drop NaNs
                logps[component] = logps[component][~np.isnan(logps[component][:])]

            plt.figure(figsize=(8,12))
            nrows = len(logps.keys())
            ncols = 2
            for row, component in enumerate(components):
                # Full range
                try:
                    col = 0
                    plt.subplot2grid((nrows,ncols),(row,col))
                    plt.hist(logps[component], bins=nbins)
                    plt.title(component)
                except Exception as e:
                    print(e)

                # Limited range
                try:
                    col = 1
                    plt.subplot2grid((nrows,ncols),(row,col))
                    plt.hist(logps[component], range=[-logP_range, +logP_range], bins=nbins)
                    plt.title(component)
                except Exception as e:
                    print(e)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    def plot_ncmc_work_old(self, filename):
        """Generate plots of NCMC work.

        Parameters
        ----------
        filename : str
            File to write PDF of NCMC work plots to.

        """
        with PdfPages(filename) as pdf:
            for envname in ['NCMCEngine', 'NCMCHybridEngine']: #self.get_environments():
                modname = envname
                work = dict()
                for direction in ['delete', 'insert']:
                    varname = '/' + modname + '/' + 'total_work_' + direction
                    try:
                        # TODO: For now, we analyze all but the last sample, so that this can be run on active simulations.
                        # Later, we should find some way to omit the last sample only if it is nonsensical.
                        work[direction] = self._ncfile[varname][:-1,:]
                        print('Found %s' % varname)
                    except Exception as e:
                        pass

                def plot_work_trajectories(pdf, work, title=""):
                    """Generate figures for the specified switching legs.
                    """
                    plt.figure(figsize=(12, 8))

                    nrows = len(work.keys())
                    ncols = 6
                    workcols = 2
                    for (row, direction) in enumerate(work.keys()):
                        #
                        # Plot work vs step
                        #

                        col = 0
                        plt.subplot2grid((nrows,ncols), (row, col), colspan=(ncols-workcols))

                        # Plot average work distribution in think solid line
                        plt.plot(work[direction].mean(0), 'k-', linewidth=1.0, alpha=1.0)
                        # Plot bundle of work trajectories in transparent lines
                        plt.plot(work[direction].T, 'k-', linewidth=0.5, alpha=0.3)
                        # Adjust axes to eliminate large-magnitude outliers (keep 98% of data in-range)
                        workvals = np.ravel(np.abs(work[direction]))
                        worklim = np.percentile(workvals, 98)
                        nsteps = work[direction].shape[1]
                        plt.axis([0, nsteps, -worklim, +worklim])
                        # Label plot
                        if row == 1: plt.xlabel('steps')
                        plt.ylabel('work / kT')
                        plt.title("%s NCMC in environment '%s' : %s" % (title, envname, direction))
                        plt.legend(['average work', 'NCMC attempts'])

                        #
                        # Plot work histogram
                        #

                        col = ncols - workcols
                        plt.subplot2grid((nrows,ncols), (row, col), colspan=workcols)

                        # Plot average work distribution in think solid line
                        #nbins = 40
                        workvals = work[direction][:-1,-1]
                        #plt.hist(workvals, nbins)
                        if workvals.std() != 0.0:
                            sns.distplot(workvals, rug=True)
                        else:
                            print('workvals has stddev of zero')
                            print(workvals)
                        # Adjust axes to eliminate large-magnitude outliers (keep 98% of data in-range)
                        #worklim = np.percentile(workvals, 98)
                        #oldaxis = plt.axis()
                        #plt.axis([-worklim, +worklim, 0, oldaxis[3]])
                        # Label plot
                        if row == 1: plt.xlabel('work / kT')
                        plt.title("total %s work" % direction)

                    plt.tight_layout()
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                if len(work) > 0:
                    # Plot work for all chemical transformations.
                    plot_work_trajectories(pdf, work, title='(all transformations)')

                    # Plot work separated out for each chemical transformation
                    #[niterations, nsteps] = work.shape
                    #transformations = dict()
                    #for iteration in range(niterations):
                    #    plot_work_trajectories(pdf, work, title='(all transformations)')
