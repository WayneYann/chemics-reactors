""" This is a file for specifying the mass flow rates and kinetic schemes for
a reactor model.  """

from pyropy import util, gas
from math import pi, sqrt
import numpy as np

# Rate function
def rate_law(T, P, vol, mw, masses, mass_fracs, eps, dp, mu):
    """
    This implements the rate law for all reactors.

    Parameters
    ----------
    T = temperature (K)
    P = pressure (kPa)
    vol = the total volume of the reactor (m^3)
    mw = list-of-lists for molecular weights (g/mol) for each species in each
        phase
    masses = a list of the total masses (kg) for each phase in the reactor
    mass_fracs = list-of-lists for the mass fractions for each species in each
        phase
    eps = list of volume fractions for each phase
    dp = list-of-lists for the particle diameters (m) for each species in each
        phase
    mu = list of dynamic viscosities (kg/m/s) for each phase

    Returns
    -------
    rates = a list/array of reaction rates for each species in every phase (kg/s)
    dH_rxn = the rate of change of the enthalpy of reaction in the reactor (J/s)
    """

    # Minimal do-nothing function (no kinetics, no heat change)
    nspec = [len(mw[i]) for i in range(len(mw))]
    rates = [np.zeros(nspec[i]) for i in range(len(nspec))]
    dH_rxn = 0.0

# Mass flow rates function
def get_mass_flow_rates(network):
        """
        This function calculates the mass flow rates between reactors in a
        network. Currently this function does nothing and is included only to
        give you an example of the function signature and to ensure that there
        is a function name in the module to overwrite if you need to.

        Parameters
        ----------
        network = a network object; this is passed automatically when the mass
            flow rates function is assigned to the network object

        Returns
        -------
        network = an updated network object
        """

    pass
