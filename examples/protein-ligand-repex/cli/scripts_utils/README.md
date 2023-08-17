Integrated retrospective and prospective analysis for perses free energy calculations.

Issue: https://github.com/choderalab/perses/issues/1136

# INPUTS
- ligands.sdf : SDF file from which the calculations are run
    - Assume it's a single file specified in the perses yaml ligand_file
    - Assume all ligands represent single protonation/tautomer/stereochemical macrostate
    - If there are multiple protonation/tautomer states that represent one real chemical compound:
        - SDTag called PARENT_LIGAND_ID that corresponds to the LIGAND_ID of the real compound, for which this microstate represents a protonation/tautomer/stereochemical state.
        - If this is a protonation/tautomer microstate it will have the following SDTags:
            - `r_epik_State_Penalty` : state penalty of this microstate in kcal/mol.
                - if no state penalty is specified, assume equal populations. This can be useful for stereoisomers.
- experimental-data.sdf/csv : Tagged experimental data for one or more ligands
    - could be the same ligands.sdf if tags are present with the following fields:
            * EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL
            * EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR
            - if csv it should have this format: SMILES, LIGAND_ID, SDData...
            - if sdf the title should match the molecule LIGAND_ID and SDDATA is sdtags
-


# OUTPUTS
- optimal-predictions.yaml : Predicted optimal binding FEs of all compounds, using ALL available experimental data.
    - Format:
        metadata:
            energy_units : kilocalories_per_mole
            temperature : <temperature in kelvin>
        ligand_data :
            ligand_id :
                absolute_binding_free_energy :
                    estimate :
                    standard_error :
                microstates : [if any ligand has multiple microstates]
                    microstate_id :
                        estimate :
                        standard_error :
                        microstate_penalty :
                        microstate_population_in_complex :
                    ...
                experimental_data:
                    measurement :
                    standard_error :
            ...
    Question: Do we also want to produce all ligand pairs with correct uncertainties?
- retrospective-optimal-reliability.yaml : Reliability indicator for optimal predictions using experimental data for everything. For each compound with exp data we use all other experimental data to compute an optimal MLE estimate.
    ligand_id :
        absolute_binding_free_energy :
            estimate :
            standard_error :
        microstates : [if any ligand has multiple microstates]
            microstate_id :
                estimate :
                standard_error :
                microstate_penalty :
                microstate_population_in_complex :
            ...
        experimental_data:
            measurement :
            standard_error :
        error :
            deviation :
            standard_error :
    ...
- retrospective-edges.yaml : Relative FE results sorted by absolute errors (magnitude) for those with experimental data. Only for edges between ligands with single microstates.
    - initial_ligand_id :
      final_ligand_id :
      absolute_error :
      signed_error :
      error_standard_error :
      calculated :
      calculated_standard_error :
      experimental :
      experimental_standard_error :
      statistics : [Optional]
    ...

- retrospective-absolutes.yaml : (For compounds with experimental data) MLE estimates of absoute binding free energies computed without experimental data, listed in order of absolute FE deviations from experiment
    ligand_id :
      absolute_error :
      signed_error :
      error_standard_error :
      calculated :
      calculated_standard_error :
      experimental :
      experimental_standard_error :
      statistics : [Optional]
      microstates : [Optional]
        microstate_id :
            estimate :
            standard_error :
            microstate_penalty :
            microstate_population_in_complex :


This script produces three major outputs:

* Optimal predictions: The predicted absolute ΔGs with uncertainties 
  (and ΔΔGs between compounds with uncertainties) integrating all available experimental data in the network
* Reliability indicator for optimal predictions: If there is more than one compound with experimental data, 
  we also provide a leave-one-out assessment of how accurately we predicted each compound with experimental data 
  by leaving its experimental free energy out of the DiffNet analysis and reporting the deviation statistics
* Retrospective debugging: For compounds with experimental data, a comparison of directly computed 
  edge ΔΔGs(without using any experimental data) with the experimental ΔΔGs, as well as how the 
  absolute ΔGs (without experimental data corrections) correlate with experimental absolute ΔGs 
  (shifting to the experimental mean in both cases)

Experimental data format:

* CSV: CSV file containing the following fields ... TODO

TODO:
* Refactor this stand-alone script into an integrated analysis class and CLI integrated into perses.
* Support experimental data in YAML format as well
