import numpy as np
import matplotlib.pyplot as plt
import sys
from openeye import oechem, oegraphsim
import logging


_logger = logging.getLogger("analysis")


class Molecule(object):
    def __init__(self, i, string):
        from perses.utils.openeye import smiles_to_oemol
        self.line = string
        details = string.split(';')
        self.index = i
        self.smiles, self.name, self.exp, self.experr, self.calc, self.calcerr = details[1:7]
        self.mol = smiles_to_oemol(self.smiles)
        self.exp = kcal_to_kt(float(self.exp))
        self.experr = kcal_to_kt(float(self.experr))
        self.calc = kcal_to_kt(float(self.calc))
        self.calcerr = kcal_to_kt(float(self.calcerr))
        self.mw = self.calculate_molecular_weight()
        self.ha = self.heavy_atom_count()
        self.simtype = None

    def calculate_molecular_weight(self):
        """ Calculates the molecular weight of an oemol

        Parameters
        ----------

        Returns
        -------
        float, molecular weight of molecule

        """
        return oechem.OECalculateMolecularWeight(self.mol)

    def heavy_atom_count(self):
        """ Counts the number of heavy atoms in an oemol

        Parameters
        ----------


        Returns
        -------
        int, number of heavy atoms in molecule
        """
        return oechem.OECount(self.mol, oechem.OEIsHeavy())

class Simulation(object):
    from simtk import unit
    def __init__(self,A,B):
        self.ligA = A
        self.ligB = B
        self.directory = f'lig{self.ligA}to{self.ligB}'
        self.vacdg = None
        self.vacddg = None
        self.soldg = None
        self.solddg = None
        self.comdg = None
        self.comddg = None

        self.vacdg_history = []
        self.soldg_history = []
        self.comdg_history = []
        self.vacddg_history = []
        self.solddg_history = []
        self.comddg_history = []

        self.count = 0
        self.load_data()

        if self.vacdg is not None and self.soldg is not None:
            self.hydrationdg = self.vacdg - self.soldg
            self.hydrationddg = (self.vacddg + self.solddg)**0.5
        else:
            print('Both vacuum and solvent legs need to be run for hydration free energies')
        if self.comdg is not None and self.soldg is not None:
            self.bindingdg = self.soldg - self.comdg
            self.bindingddg = (self.comddg + self.solddg)**0.5
        else:
            print('Both solvent and complex legs need to be run for binding free energies')

    @staticmethod
    def _kt_to_kcal(x):
        q = unit.quantity.Quantity(x, unit = unit.kilojoules_per_mole)
        return q.in_units_of(unit.kilocalories_per_mole)._value

    def load_data(self):
        """ Calculate relative free energy details from the simulation by performing MBAR on the vacuum and solvent legs of the simualtion.

        Parameters
        ----------

        Returns
        -------
        None

        """
        from pymbar import timeseries
        from pymbar import MBAR
        from perses.analysis import utils
        import os
        from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer

        # find the output files
        output = [x for x in os.listdir(self.directory) if x[-3:] == '.nc' and 'checkpoint' not in x]


        for out in output:
            if 'vacuum' in out:
                vacuum_reporter = MultiStateReporter(f'{self.directory}/{out}')
                vacuum_analyzer = MultiStateSamplerAnalyzer(vacuum_reporter)
                f_ij, df_ij = vacuum_analyzer.get_free_energy()
                self.vacdg = _kt_to_kcal(f_ij[0, -1])
                self.vacddg = _kt_to_kcal(df_ij[0, -1] ** 2)
                self.vacf_ij = f_ij
                self.vacdf_ij = df_ij
            elif'solvent' in out:
                solvent_reporter = MultiStateReporter(f'{self.directory}/{out}')
                solvent_analyzer = MultiStateSamplerAnalyzer(solvent_reporter)
                f_ij, df_ij = solvent_analyzer.get_free_energy()
                self.soldg = _kt_to_kcal(f_ij[0, -1])
                self.solddg = _kt_to_kcal(df_ij[0, -1] ** 2)
                self.solf_ij = f_ij
                self.soldf_ij = df_ij
            elif 'complex' in out:
                complex_reporter = MultiStateReporter(f'{self.directory}/{out}')
                complex_analyzer = MultiStateSamplerAnalyzer(complex_reporter)
                f_ij, df_ij = complex_analyzer.get_free_energy()
                self.comdg = _kt_to_kcal(f_ij[0, -1])
                self.comddg = _kt_to_kcal(df_ij[0, -1] ** 2)
                self.comf_ij = f_ij
                self.comdf_ij = df_ij
        return


    def historic_fes(self,stepsize=100):
        from pymbar import timeseries
        from pymbar import MBAR
        from perses.analysis import utils
        import os
        from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer

        # find the output files
        output = [x for x in os.listdir(self.directory) if x[-3:] == '.nc' and 'checkpoint' not in x]

        for out in output:
            if 'vacuum' in out:
                vacuum_reporter = MultiStateReporter(f'{self.directory}/{out}')
                ncfile = utils.open_netcdf(f'{self.directory}/{out}')
                n_iterations = ncfile.variables['last_iteration'][0]
                for step in range(stepsize, n_iterations, stepsize):
                    vacuum_analyzer = MultiStateSamplerAnalyzer(vacuum_reporter,max_n_iterations=step)
                    f_ij, df_ij = vacuum_analyzer.get_free_energy()
                    self.vacdg_history.append(_kt_to_kcal(f_ij[0, -1]))
                    self.vacddg_history.append(_kt_to_kcal(df_ij[0,-1]))
            if 'solvent' in out:
                solvent_reporter = MultiStateReporter(f'{self.directory}/{out}')
                ncfile = utils.open_netcdf(f'{self.directory}/{out}')
                n_iterations = ncfile.variables['last_iteration'][0]
                for step in range(stepsize, n_iterations, stepsize):
                    solvent_analyzer = MultiStateSamplerAnalyzer(solvent_reporter,max_n_iterations=step)
                    f_ij, df_ij = solvent_analyzer.get_free_energy()
                    self.soldg_history.append(_kt_to_kcal(f_ij[0, -1]))
                    self.solddg_history.append(_kt_to_kcal(df_ij[0,-1]))
            if 'complex' in out:
                complex_reporter = MultiStateReporter(f'{self.directory}/{out}')
                ncfile = utils.open_netcdf(f'{self.directory}/{out}')
                n_iterations = ncfile.variables['last_iteration'][0]
                for step in range(stepsize, n_iterations, stepsize):
                    complex_analyzer = MultiStateSamplerAnalyzer(complex_reporter,max_n_iterations=step)
                    f_ij, df_ij = complex_analyzer.get_free_energy()
                    self.comdg_history.append(_kt_to_kcal(f_ij[0, -1]))
                    self.comddg_history.append(_kt_to_kcal(df_ij[0,-1]))
        return


    def sample_history(self,method='binding'):
        vac = self.vacdg_history[self.count]
        sol = self.soldg_history[self.count]
        com = self.comdg_history[self.count]
        vacvar = self.vacddg_history[self.count]
        solvar = self.solddg_history[self.count]
        comvar = self.comddg_history[self.count]

        self.count += 1

        if method == 'binding':
            return sol - com , (solvar**2 + comvar**2)**0.5
        elif method == 'hydration':
            return  vac - sol , (solvar**2 + vacvar**2)**0.5
        else:
            print('method not recognised, choose binding or hydration')

    def reset_history(self):
        self.count = 0


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
    import itertools
    import os

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
