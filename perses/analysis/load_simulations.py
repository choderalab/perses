import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import sys
from perses.analysis import utils
from openeye import oechem, oegraphsim
#from perses.utils import openeye
from openmoltools import openeye
import logging


from pymbar import timeseries
from pymbar import MBAR

_logger = logging.getLogger("analysis")


class molecule(object):
    def __init__(self, i, string):
        self.line = string
        details = string.split(';')
        self.index = i
        self.smiles, self.name, self.exp, self.experr, self.calc, self.calcerr = details[1:7]
        self.mol = openeye.smiles_to_oemol(self.smiles)
        self.exp = kcal_to_kt(float(self.exp))
        self.experr = kcal_to_kt(float(self.experr))
        self.calc = kcal_to_kt(float(self.calc))
        self.calcerr = kcal_to_kt(float(self.calcerr))
        self.mw = self.calc_mw()
        self.ha = self.heavy_atom_count()

    def calc_mw(self):
        return oechem.OECalculateMolecularWeight(self.mol)

    def heavy_atom_count(self):
        return oechem.OECount(self.mol, oechem.OEIsHeavy())


class simulation(object):
    def __init__(self,A,B):
        self.ligA = A
        self.ligB = B
        self.directory = f'lig{self.ligA}to{self.ligB}'
        self.load_data()

        if self.sdg is not None and self.vdg is not None:
            self.dg = self.vdg - self.sdg
            self.dg_err = (self.vvariance + self.svariance)**0.5
        else:
            self.dg = None
            self.dg_err = None

    def load_data(self):
        """ Calculate relative free energy details from the simulation by performing MBAR on the vacuum and solvent legs of the simualtion.

        Parameters
        ----------

        Returns
        -------
        None

        """
        # TODO need to adapt for either complex/solvent or solvent/vacuum or both
        # find the output files
        output = [x for x in os.listdir(self.directory) if x[-3:] == '.nc' and 'checkpoint' not in x]

        if len(output) != 2:
            print('not both legs done')
            self.sdg = None
            self.vdg = None
        else:
            for out in output:
                if 'vacuum' in out:
                    ncfile = utils.open_netcdf(self.directory+'/'+out)
                    energies = np.array(ncfile.variables['energies'])
                    states = np.array(ncfile.variables['states'])
                    mbar, Neff, t0 = self.compute_mbar(energies, states)
                    f_ij, df_ij, theta = mbar.getFreeEnergyDifferences()
                    self.vdg = f_ij[0, -1]
                    self.vvariance = df_ij[0, -1] ** 2
                    self.vefficiencies = df_ij[0, -1] ** -2
                    self.vnsample = Neff
                    self.vf_k = f_ij[0, :]
                    self.vdf_k = df_ij[0, :]
                    self.vt0 = t0
                if 'solvent' in out:
                    ncfile = utils.open_netcdf(self.directory+'/'+out)
                    energies = np.array(ncfile.variables['energies'])
                    states = np.array(ncfile.variables['states'])
                    mbar, Neff, t0 = self.compute_mbar(energies, states)
                    f_ij, df_ij, theta = mbar.getFreeEnergyDifferences()
                    self.sdg = f_ij[0, -1]
                    self.svariance = df_ij[0, -1] ** 2
                    self.sefficiencies = df_ij[0, -1] ** -2
                    self.snsample = Neff
                    self.sf_k = f_ij[0, :]
                    self.sdf_k = df_ij[0, :]
                    self.st0 = t0
        return

    def compute_mbar(self, energies, states):
        """ Compute MBAR free energy, number of effective states and t0 for a simulation leg

        Parameters
        ----------
        energies : np.array
            array of energies for each state and iteration of the simulation
        states : np.array
            array of states visited during simulation

        Returns
        -------
        mbar : pymbar.MBAR

        Neff_max : int
            number of effective states

        t0 : int
            identified iteration of equilibration

        """
        [niterations, nreplicas, nstates] = energies.shape
        if niterations == 1:
            print(f'Simulation {self.directory} has only one iteration step')
            return

        # Form u_n
        u_n = np.zeros([niterations])
        for iteration in range(niterations):
            for replica in range(nreplicas):
                state = states[iteration, replica]
                u_n[iteration] += energies[iteration, replica, state]

        # Detect equilibration
        [t0, g, Neff_max] = timeseries.detectEquilibration(u_n)
        series = timeseries.subsampleCorrelatedData(np.arange(t0, niterations), g=g)
        indices = t0 + series

        # Extract arrays
        u_kn = np.zeros([nstates, nreplicas * len(indices)])
        N_k = np.zeros([nstates], np.int32)
        for replica in range(nreplicas):
            u_kn[:, (replica * len(indices)):(replica + 1) * len(indices)] = energies[indices, replica, :].transpose()
            replica_N_k, edges = np.histogram(states[indices, replica], bins=np.arange(nstates + 1))
            N_k += replica_N_k

        mbar = MBAR(u_kn, N_k)
        return mbar, Neff_max, t0


    def historic_fes(self,stepsize=10):
        """ Calculate free energies with incremental sections of the simulation, to generate free energy samples that can be post-proccessed in adaptive sampling schemes

        Parameters
        ----------
        stepsize : int, default = 10
            Number of simulation steps between each energy evaluation

        Returns
        -------
        list, list
            list of the relative free energies, and the variances, of length (# steps in simulations)/(stepsize)

        """
        vhistory = []
        shistory = []
        dvhistory = []
        dshistory = []
        output = [x for x in os.listdir(self.directory) if x[-3:] == '.nc' and 'checkpoint' not in x]

        if len(output) != 2:
            print('not both legs done')
        else:
            for out in output:
                if 'vacuum' in out:
                    ncfile = utils.open_netcdf(self.directory+'/'+out)
                    energies = np.array(ncfile.variables['energies'])
                    states = np.array(ncfile.variables['states'])
                    niterations, _, _ = energies.shape
                    for x in range(stepsize, niterations, stepsize):
                        energies_slice = energies[:x][:][:]
                        states_slice = states[:x][:][:]
                        mbar, Neff, t0 = self.compute_mbar(energies_slice, states_slice)
                        f_ij, df_ij, theta = mbar.getFreeEnergyDifferences()
                        vhistory.append(f_ij[0, -1])
                        dvhistory.append(df_ij[0, -1] ** 2)
                if 'solvent' in out:
                    ncfile = utils.open_netcdf(self.directory+'/'+out)
                    energies = np.array(ncfile.variables['energies'])
                    states = np.array(ncfile.variables['states'])
                    for x in range(stepsize, niterations, stepsize):
                        energies_slice = energies[:x][:][:]
                        states_slice = states[:x][:][:]
                        mbar, Neff, t0 = self.compute_mbar(energies_slice, states_slice)
                        f_ij, df_ij, theta = mbar.getFreeEnergyDifferences()
                        shistory.append(f_ij[0, -1])
                        dshistory.append(df_ij[0, -1] ** 2)

        history = []
        variance = []
        for i in range(len(vhistory)):
            history.append(vhistory[i] - shistory[i])
            variance.append((dvhistory[i]**2 + dshistory[i]**2)**0.5)
        return history, variance

def kcal_to_kt(x):
    """
    This should be deleted and just use the simtk units protocol
    :param x: float, energy in kcal
    :return: float, energy in kT
    """
    # TODO remove this
    return x*1.688

def get_experimental(molecules, i,j):
    """ Determine experimental relative free energy from two experimntal absolute results

    Parameters
    ----------
    molecules : list
        list of load_simulation.molecule objects
    i : int
        index of first molecule
    j : int
        index of second molecule

    Returns
    -------
    tuple
        relative free energy and associated error

    """
    moli = molecules[i]
    molj = molecules[j]
    ddG = moli.exp - molj.exp
    ddG_err = moli.experr - molj.experr
    return (ddG, ddG_err)


def load_experimental(exp_file):
    """ Load details from a freesolv database.txt-like file

    Parameters
    ----------
    exp_file : str
        path to text file

    Returns
    -------
    list
        list of load_simulation.molecule objects, contained in the textfile

    """
    molecules = []
    with open(exp_file) as f:
        for i, line in enumerate(f):
            molecules.append(molecule(i, line))
    return molecules

def run(molecules,simtype='sams',offline_freq=10):
    """Load the simulation data for a set of molecules, for both forward and backward simulations

    Parameters
    ----------
    molecules : list
        list of load_simulation.molecule objects, of which to find simulation data for
    simtype : type
        Description of parameter `simtype`.
    offline_freq : type
        Description of parameter `offline_freq`.

    Returns
    -------
    type
        Description of returned object.

    """

    n_ligands = len(molecules)
    all_simulations = []
    for a, b in itertools.combinations(range(0, n_ligands), 2):
        path = f'lig{a}to{b}'
        if os.path.isdir(path) == True:
            sim = simulation(a, b)
            all_simulations.append(sim)
        else:
            print(f'Output directory lig{a}to{b} doesnt exist')

        # now run the opposite direction
        path = f'lig{b}to{a}'
        if os.path.isdir(path) == True:
            sim = simulation(b, a)
            all_simulations.append(sim)
        else:
            print(f'Output directory lig{b}to{a} doesnt exist')

    return all_simulations

if __name__ == '__main__':
    run(sys.argv[1])
