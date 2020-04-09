#!/usr/bin/env python

from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from simtk import unit
import numpy as np
import simtk.openmm as openmm
from openmmtools.constants import kB
from openmmtools import utils


# In[119]:


class LoopyAlchemicalNonequilibriumLangevinIntegrator(AlchemicalNonequilibriumLangevinIntegrator):
    """
    Subclass of `AlchemicalNonequilibriumLangevinIntegrator` that hardcodes the following integration scheme:

    Step 1: lambda_0 equilibrium sampling
        run nsteps_eq of BAOAB (V R O R V) integration at the alchemical lambda_0 endstate
    Step 2: Forward Annealing
        run n_steps_neq of BAOAB (V R O R V H); 'H': Hamiltonian update step
    Step 3: lambda_1 equilibrium sampling
        run Step 1 at the lambda_1 endstate
    Step 3: Reverse Annealing
        run Step 2 in the reverse direction

    forward and backward works are saved to `forward_work` and `backward_work`, respectively
    """
    def __init__(self,
                 alchemical_functions = None,
                 nsteps_eq = 1000,
                 nsteps_neq = 100,
                 **kwargs):
        """        
        arguments
            nsteps_eq : int
                number of equilibration steps to run at either endstate
            nsteps_neq : int
                number of nonequilibrium annealing steps to run in either direction

        parameters
            _n_steps_eq : int
                number of BAOAB loops to run at each endstate in an attempt to generate i.i.d samples from endstates
        """
        self._n_steps_eq = nsteps_eq
        splitting = 'V R O R V'
        super().__init__(alchemical_functions, splitting, nsteps_neq = nsteps_neq, **kwargs)
    
    def _add_global_variables(self):
        """
        modify the super (_add_global_variables) to add a collector for forward and reverse annealing, as well as to specify the number of equilibration
        steps at endstates.
        """
        super()._add_global_variables()
        self.addGlobalVariable('forward_work', 0)
        self.addGlobalVariable('backward_work', 0)
        self.addGlobalVariable('n_steps_eq', self._n_steps_eq)
        self.addGlobalVariable('eq_step', 0)
        
        #overwrite because this is set to 0 in super() (n_H = 1, but this is omitted in the splitting string);
        #see https://github.com/choderalab/openmmtools/blob/c2b61c410b255c4e08927acf8cfcb1cf46f64b70/openmmtools/integrators.py#L1818
        self.setGlobalVariableByName('n_lambda_steps', self._n_steps_neq) 
    
    def _add_integrator_steps(self):
        """
        hardcode a custom integration scheme specified at the top of the class
        """
        #init/reset
        self.addConstrainPositions() #constrain positions
        self.addConstrainVelocities() #and velocities
        self._add_reset_step() #reset lambda, protocol_works, heat, shadow, ghmc, step (unused), lambda_step, and alchemical_params
        
        #lambda_0 equilibration
        self.beginWhileBlock("eq_step < n_steps_eq")
        self._add_canonical_integrator_steps() #add VRORV
        self.addComputeGlobal('eq_step', "eq_step + 1") #increment eq_step
        self.endBlock()
        
        #forward anneal
        self.beginWhileBlock("lambda_step < n_lambda_steps") #anneal forward until lambda step is n_lambda_steps (lambda = lambda_step / n_lambda_steps)
        self._add_canonical_integrator_steps() #add VRORV
        self._substep_function('H') #add the H step...
        self.endBlock()
        self.addComputeGlobal('forward_work', 'protocol_work') #log the forward protocol work
        self.addComputeGlobal('protocol_work', '0') #reset the protocol work for the reverse annealing
        
        
        #lambda_1 equilibration
        self.addComputeGlobal('eq_step', '0') #reset eq_step counter
        self.beginWhileBlock("eq_step < n_steps_eq") 
        self._add_canonical_integrator_steps() #add VRORV
        self.addComputeGlobal('eq_step', "eq_step + 1") #increment eq_step
        self.endBlock()
        
        #reverse anneal; don't reset lambda (it is annealing in backward direction)
        self.beginWhileBlock("lambda_step > 0") #anneal backward until lambda step is 0 again
        self._add_canonical_integrator_steps() #add VRORV
        self._substep_function('L') #add the L step (backward H)
        self.endBlock()
        self.addComputeGlobal('backward_work', 'protocol_work')
    
    def _add_reset_step(self):
        """
        Reset the alchemical lambda to its starting value (lambda = 0)
        """
        self.addComputeGlobal("lambda", "0")
        self.addComputeGlobal("step", "0")
        self.addComputeGlobal("lambda_step", "0")
        self.addComputeGlobal('forward_work', "0")
        self.addComputeGlobal('backward_work', "0")
        self.addComputeGlobal('protocol_work', '0')
        self.addComputeGlobal('eq_step', '0')
        # Add all dependent parameters
        self._add_update_alchemical_parameters_step()
    
    def _add_canonical_integrator_steps(self):
        """
        add the BAOAB integrator
        """
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})
        
        for i, step in enumerate(self._splitting.split()): #splitting is just 'V R O R V'
            self._substep_function(step)
        
    @property
    def _step_dispatch_table(self):
        """
        add an L step to the dispatch table
        """
        dispatch_table = super()._step_dispatch_table
        dispatch_table['L'] = (self._add_backward_alchemical_perturbation_step, False)
        return dispatch_table
    
    def _add_backward_alchemical_perturbation_step(self):
        """
        _add_alchemical_perturbation_step except the lambda step is decremented
        """
        # Store initial potential energy
        self.addComputeGlobal("Eold", "energy")

        # Update lambda and increment that tracks updates.
        self.addComputeGlobal('lambda', '(lambda_step-1)/n_lambda_steps')
        self.addComputeGlobal('lambda_step', 'lambda_step - 1') #decrement instead of increment

        # Update all slaved alchemical parameters
        self._add_update_alchemical_parameters_step()

        # Accumulate protocol work
        self.addComputeGlobal("Enew", "energy")
        self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)")
        

