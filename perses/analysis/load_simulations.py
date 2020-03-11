import numpy as np
import matplotlib.pyplot as plt
import sys
from openeye import oechem, oegraphsim

import logging
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_logger = logging.getLogger("load_simulations")


class Simulation(object):
    """Object that holds the results of a free energy simulation.
    # TODO make this much more flexible
    Assumes that the output is in the form lig{A}to{B}/out-{phase}.nc

    This automatically loads the data and performs the data. Results can be accessed through attributes.

    >>> # Load and analyze the simulation for `lig0to1/out.*`
    >>> simulation_results = Simulation(0,1)
    >>> # Report the binding free energy
    >>> print(f'Relative binding free energy: {simulation_results.bindingdg}')


    Parameters
    ----------
    A : int
        integer of first ligand
    B : int
        integer of second ligand

    Attributes
    ----------
    ligA : int
        integer of first ligand
    ligB : int
        integer of first ligand
    directory : string
        output directory location : lig{A}to{B}
    vacdg : float
        relative free energy in vacuum phase (kcal/mol)
    vacddg : float
        std error in relative free energy in vacuum phase (kcal/mol)
    soldg : float
        relative free energy in solvent phase (kcal/mol)
    solddg : float
        std error in relative free energy in solvent phase (kcal/mol)
    comdg : float
        relative free energy in complex phase (kcal/mol)
    comddg : float
        std error in relative free energy in complex phase (kcal/mol)
    vacdg_history : list(float)
        vacuum free energy at equally spaced intervals of simulation (kcal/mol)
    soldg_history : type
        solvent free energy at equally spaced intervals of simulation (kcal/mol)
    comdg_history : type
        complex free energy at equally spaced intervals of simulation (kcal/mol)
    vacddg_history : type
        std error in vacuum free energy at equally spaced intervals of simulation (kcal/mol)
    solddg_history : type
        std error in solvent free energy at equally spaced intervals of simulation (kcal/mol)
    comddg_history : type
        std error in complex free energy at equally spaced intervals of simulation (kcal/mol)
    count : int
        Number of times the 'histories' have been 'sampled' from
    hydrationdg : float
        relative hydration free energy (kcal/mol)
    hydrationddg : float
        std error in relative hydration free energy (kcal/mol)
    bindingdg : float
        relative hydration free energy (kcal/mol)
    bindingddg : float
        std error in relative hydration free energy (kcal/mol)

    """
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
        self._load_data()

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
        q = unit.quantity.Quantity(x, unit=unit.kilojoules_per_mole)
        return q / unit.kilocalories_per_mole

    def _load_data(self):
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


    def historic_fes(self, stepsize=100):
        """ Function that performs mbar at intervals of the simulation
        by postprocessing. Can be slow if stepsize is small

        Parameters
        ----------
        stepsize : int, optional, default=100
            number of iterations at which to run MBAR

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
        """Get the free energy from the historic FE's in steps

        Parameters
        ----------
        method : string, default='binding'
            either 'binding' or 'hydration' to get that free energy

        Returns
        -------
        float, float
            the relative free energy and it's associated std error (kcal/mol)

        """
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
