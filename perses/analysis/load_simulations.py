import numpy as np
import matplotlib.pyplot as plt
import sys
from openeye import oechem, oegraphsim
from openmmtools.constants import kB

import logging
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_logger = logging.getLogger("load_simulations")


class Simulation(object):
    """Object that holds the results of a free energy simulation.
    # TODO make this much more flexible
    Assumes that the output is in the form lig{A}to{B}/out-{phase}.nc

    This automatically loads the data and performs the data. Results can be accessed through attributes.

    >>> # Load and analyze the simulation for `lig0to1/*.nc`
    >>> simulation_results = Simulation(0,1)
    >>> # Report the binding free energy
    >>> print(f'Relative binding free energy: {simulation_results.bindingdg}')


    Parameters
    ----------
    directory : string
        path to output directory of simulation.
        this is the `trajectory_directory` variable in
        a perses simulation yaml file

    Attributes
    ----------
    directory : string
        output directory location : lig{A}to{B}
    _vacdg : float
        relative free energy in vacuum phase (kcal/mol)
    _vacddg : float
        std error in relative free energy in vacuum phase (kcal/mol)
    _soldg : float
        relative free energy in solvent phase (kcal/mol)
    _solddg : float
        std error in relative free energy in solvent phase (kcal/mol)
    _comdg : float
        relative free energy in complex phase (kcal/mol)
    _comddg : float
        std error in relative free energy in complex phase (kcal/mol)
    _vacdg_history : list(float)
        vacuum free energy at equally spaced intervals of simulation (kcal/mol)
    _soldg_history : type
        solvent free energy at equally spaced intervals of simulation (kcal/mol)
    _comdg_history : type
        complex free energy at equally spaced intervals of simulation (kcal/mol)
    _vacddg_history : type
        std error in vacuum free energy at equally spaced intervals of simulation (kcal/mol)
    _solddg_history : type
        std error in solvent free energy at equally spaced intervals of simulation (kcal/mol)
    _comddg_history : type
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

    def __init__(self, directory):
        self.directory = directory
        self._vacdg = None
        self._vacddg = None
        self._soldg = None
        self._solddg = None
        self._comdg = None
        self._comddg = None
        self.hydrationdg = None
        self.hydrationddg = None
        self.bindingdg = None
        self.bindingddg = None


        self._vacdg_history = []
        self._soldg_history = []
        self._comdg_history = []
        self._vacddg_history = []
        self._solddg_history = []
        self._comddg_history = []

        self.count = 0

        self._load_data()

        if self._vacdg is not None and self._soldg is not None:
            self.hydrationdg = self._vacdg - self._soldg
            self.hydrationddg = (self._vacddg**2 + self._solddg**2)**0.5
        else:
            print('Both vacuum and solvent legs need to be run for hydration free energies')
        if self._comdg is not None and self._soldg is not None:
            self.bindingdg = self._soldg - self._comdg
            self.bindingddg = (self._comddg**2 + self._solddg**2)**0.5
        else:
            print('Both solvent and complex legs need to be run for binding free energies')

    def _load_data(self):
        """ Calculate relative free energy details from the simulation by performing MBAR on the vacuum and solvent legs of the simualtion.

        Parameters
        ----------

        Returns
        -------
        None

        """
        from pymbar import MBAR
        from perses.analysis import utils
        import os
        from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer
        from simtk import unit

        # find the output files
        output = [x for x in os.listdir(self.directory) if x[-3:] == '.nc' and 'checkpoint' not in x]

        for out in output:
            if 'vacuum' in out:
                vacuum_reporter = MultiStateReporter(f'{self.directory}/{out}')
                vacuum_analyzer = MultiStateSamplerAnalyzer(vacuum_reporter)
                f_ij, df_ij = vacuum_analyzer.get_free_energy()
                f = f_ij[0,-1] * vacuum_analyzer.kT
                self._vacdg = f.in_units_of(unit.kilocalories_per_mole)
                df = df_ij[0, -1] * vacuum_analyzer.kT
                self._vacddg = df.in_units_of(unit.kilocalories_per_mole)
                self._vacf_ij = f_ij
                self._vacdf_ij = df_ij
            elif'solvent' in out:
                solvent_reporter = MultiStateReporter(f'{self.directory}/{out}')
                solvent_analyzer = MultiStateSamplerAnalyzer(solvent_reporter)
                f_ij, df_ij = solvent_analyzer.get_free_energy()
                f = f_ij[0,-1] * solvent_analyzer.kT
                self._soldg = f.in_units_of(unit.kilocalories_per_mole)
                df = df_ij[0, -1] * solvent_analyzer.kT
                self._solddg = df.in_units_of(unit.kilocalories_per_mole)
                self._solf_ij = f_ij
                self._soldf_ij = df_ij
            elif 'complex' in out:
                complex_reporter = MultiStateReporter(f'{self.directory}/{out}')
                complex_analyzer = MultiStateSamplerAnalyzer(complex_reporter)
                f_ij, df_ij = complex_analyzer.get_free_energy()
                f = f_ij[0,-1] * complex_analyzer.kT
                self._comdg = f.in_units_of(unit.kilocalories_per_mole)
                df = df_ij[0, -1] * complex_analyzer.kT
                self._comddg = df.in_units_of(unit.kilocalories_per_mole)
                self._comf_ij = f_ij
                self._comdf_ij = df_ij
        return

    def report(self):
        if self.bindingdg is not None:
            print(f'Relative binding free energy is {self.bindingdg} ({self.bindingddg})')
        if self.hydrationdg is not None:
            print(f'Relative binding free energy is {self.hydrationdg} ({self.hydrationddg})')

    # TODO fix this whole thing to be consistent with above
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
                    self._vacdg_history.append(_kt_to_kcal(f_ij[0, -1]))
                    self._vacddg_history.append(_kt_to_kcal(df_ij[0,-1]))
            if 'solvent' in out:
                solvent_reporter = MultiStateReporter(f'{self.directory}/{out}')
                ncfile = utils.open_netcdf(f'{self.directory}/{out}')
                n_iterations = ncfile.variables['last_iteration'][0]
                for step in range(stepsize, n_iterations, stepsize):
                    solvent_analyzer = MultiStateSamplerAnalyzer(solvent_reporter,max_n_iterations=step)
                    f_ij, df_ij = solvent_analyzer.get_free_energy()
                    self._soldg_history.append(_kt_to_kcal(f_ij[0, -1]))
                    self._solddg_history.append(_kt_to_kcal(df_ij[0,-1]))
            if 'complex' in out:
                complex_reporter = MultiStateReporter(f'{self.directory}/{out}')
                ncfile = utils.open_netcdf(f'{self.directory}/{out}')
                n_iterations = ncfile.variables['last_iteration'][0]
                for step in range(stepsize, n_iterations, stepsize):
                    complex_analyzer = MultiStateSamplerAnalyzer(complex_reporter,max_n_iterations=step)
                    f_ij, df_ij = complex_analyzer.get_free_energy()
                    self._comdg_history.append(_kt_to_kcal(f_ij[0, -1]))
                    self._comddg_history.append(_kt_to_kcal(df_ij[0,-1]))
        return

    def sample_history(self, method='binding'):
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
        vac = self._vacdg_history[self.count]
        sol = self._soldg_history[self.count]
        com = self._comdg_history[self.count]
        vacvar = self._vacddg_history[self.count]
        solvar = self._solddg_history[self.count]
        comvar = self._comddg_history[self.count]

        self.count += 1

        if method == 'binding':
            return sol - com, (solvar**2 + comvar**2)**0.5
        elif method == 'hydration':
            return vac - sol, (solvar**2 + vacvar**2)**0.5
        else:
            print('method not recognised, choose binding or hydration')

    def reset_history(self):
        self.count = 0
