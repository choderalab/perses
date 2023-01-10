"""
Module for the analysis of results of non-equilibrium cycling simulations.
"""


def compute_free_energy(forward_accumulated, reverse_accumulated):
    """
    Computes the free energy estimate using MBAR.

    Parameters
    ----------
    forward_accumulated: float
        Cumulated work for the forward transformation.
    reverse_accumulated: float
        Cumulated work for the reverse transformation.

    Returns
    -------
    free_energy: float
        Free energy estimate using MBAR.
    error: float
        Error for the free energy estimate using MBAR.
    """
    import pymbar
    # Compute dg, ddg
    dg, ddg = pymbar.bar.BAR(forward_accumulated, reverse_accumulated)
    return dg, ddg


def bootstrap_free_energy(d_works):
    """
    Computes free energy estimates using bootstrapping.
    Parameters
    ----------
    d_works: dict
        Nested dictionary with phases as keys and transformation direction as inner keys.

    Returns
    -------
    d_dgs: dict
        Dictionary with the free energy estimates of a given phase.
    binding_dg: float
        Absolute free energy estimate.
    """
    d_dgs_phase = {}
    complex_dg, complex_ddg = compute_free_energy(d_works['complex']['forward'], d_works['complex']['reverse'])
    apo_dg, apo_ddg = compute_free_energy(d_works['apo']['forward'], d_works['apo']['reverse'])
    d_dgs_phase['complex'] = (complex_dg, complex_ddg)
    d_dgs_phase['apo'] = (apo_dg, apo_ddg)

    binding_dg = complex_dg - apo_dg
    #     binding_ddg = (apo_ddg**2 + complex_ddg**2)**0.5
    return d_dgs_phase, binding_dg
