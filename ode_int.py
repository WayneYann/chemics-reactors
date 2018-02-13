"""This module implements the actual ODE integrator. It currently wraps the
SciPy-wrapped version of the ODE integrator VODE, but others could be
implemented."""
import reactor_network as rn
import numpy as np
from scipy.integrate import ode

def vode_y(t, network):
    """
    This function extracts the y vector as a function of time in a form
    compatible with the ODE solver vode.

    Parameters
    ----------
    t = the simulation time (s) (not used explicitly)
    *args = a tuple of additional arguments with information for evaluating the
        balance equations -- only needs a network model at the moment

    Returns
    -------
    y = an array of the independent variables (mass fractions, temperatures,
        etc.)
    """

    # Update the mass fractions, etc. in the reactor objects
    current_eq_idx = 0

    # Create a list of all the reactor nodes in the network
    reactor_nodes = list(node_idx for node_idx in range(len(network.nodes)) if
        isinstance(network.nodes[node_idx], rn.Reactor))

    # Initialize the y vector
    y = []

    # Get the total masses for each phase in the reactor
    for node_idx in reactor_nodes:
        num_phases = network.nodes[node_idx].num_phases
        y.extend(network.nodes[node_idx].masses)
        current_eq_idx += num_phases

    # Get the species mass fractions for each reactor and phase
    for node_idx in reactor_nodes:
        num_phases = network.nodes[node_idx].num_phases
        for phase_idx in range(num_phases):
            num_species = len(network.nodes[node_idx].mass_fracs[phase_idx])
            y.extend(network.nodes[node_idx].mass_fracs[phase_idx])
            current_eq_idx += num_species

    # Update the temperatures and masses for each reactor
    for node_idx in reactor_nodes:
        y.extend([network.nodes[node_idx].T])
        current_eq_idx += 1

    return np.array(y)

def vode_rhs(t, y, network):
    """
    This function computes the derivative of the system of ODEs for use with the
    ODE solver vode. The functions should be of the form y' = f(t, y).

    Parameters
    ----------
    t = the simulation time (s) (not used explicitly)
    y = an array of the independent variables (mass fractions, temperatures,
        etc.)
    *args = a series of additional arguments with information for evaluating the
        balance equations -- only needs a network model at the moment

    Returns
    -------
    yprime = an array of the derivatives (the RHS of the ODEs)
    """

    # Update the mass fractions, etc. in the reactor objects
    current_eq_idx = 0

    # Allocate memory for yprime
    yprime = np.zeros(y.size)

    # Create a list of all the reactor nodes in the network
    reactor_nodes = list(node_idx for node_idx in range(len(network.nodes)) if
        isinstance(network.nodes[node_idx], rn.Reactor))

    # Skip the total masses for each phase in the reactor -- these are
    # calculated automatically based on the temperature, pressure, and
    # composition
    for node_idx in reactor_nodes:
        num_phases = network.nodes[node_idx].num_phases
        current_eq_idx += num_phases

    # Update the species mass fractions for each reactor and phase
    for node_idx in reactor_nodes:
        num_phases = network.nodes[node_idx].num_phases
        for phase_idx in range(num_phases):
            num_species = len(network.nodes[node_idx].mass_fracs[phase_idx])
            network.nodes[node_idx].mass_fracs[phase_idx] = (
                y[current_eq_idx:current_eq_idx+num_species])
            current_eq_idx += num_species

    # Update the temperatures and masses for each reactor
    for node_idx in reactor_nodes:
        network.nodes[node_idx].T = y[current_eq_idx]
        network.nodes[node_idx].get_phase_masses()
        current_eq_idx += 1

    # Update the flow rates in the network
    network.get_mass_flow_rates()

    # For each node calculate the mass source terms (per-species and per-phase)
    for node_idx in reactor_nodes:
        network.nodes[node_idx].get_species_rates()

    # Evaluate the RHS expressions

    # Index to track where we are in the overall system of equations as we fill
    # in the RHS array
    current_eq_idx = 0

    # Overall continuity equations for each phase and node
    for node_idx in reactor_nodes:
        reactor_id = network.nodes[node_idx].node_id
        num_phases = network.nodes[node_idx].num_phases
        node_mass_change = network.get_mass_change(reactor_id)
        yprime[current_eq_idx:current_eq_idx+num_phases] = node_mass_change
        current_eq_idx += num_phases

    # Species equations for each phase and node
    for node_idx in reactor_nodes:
        reactor_id = network.nodes[node_idx].node_id
        num_phases = network.nodes[node_idx].num_phases
        for phase_idx in range(num_phases):
            num_species = len(network.nodes[node_idx].mass_fracs[phase_idx])
            mass_frac_change = network.get_species_mass_frac_change(reactor_id,
                phase_idx)
            yprime[current_eq_idx:current_eq_idx+num_species] = mass_frac_change
            current_eq_idx += num_species

    # Energy equations (one for each node)
    for node_idx in reactor_nodes:
        reactor_id = network.nodes[node_idx].node_id
        yprime[current_eq_idx] = network.get_T_change(reactor_id)
        current_eq_idx += 1

    # We're done
    return yprime

class ode_integrator():
    """
    This is a convenience class wrapping a variety of different ODE/DAE solvers
    into a common interface.
    """

    def __init__(self, solver, settings, network):
        """
        Initialize the specified solver with the given settings.

        Parameters
        ----------
        solver = a string specifying the solver to use (currently only 'vode' is
            allowed)
        settings = a dictionary with the settings specifice to the selected
            solver (passed unchanged to the initializer for the actual solver)
        network = a network object containing all the information about the
            system

        Returns
        -------
        self = an object with the system of ODEs/DAEs associated with the chosen
            solver
        """

        if solver == 'vode':
            self.solver = solver
            self.ode_obj = ode(vode_rhs, jac=None)
            self.ode_obj.set_integrator(solver, **settings)
        else:
            raise NotImplementedError('Invalid ODE/DAE solver: ' + solver)
        self.network = network

    def set_ic(self, t0, reset=False):
        """
        This function sets the initial conditions for the ODE solver.

        Parameters
        ----------
        t0 = the start time
        reset = a Boolean indicating whether we want to reset the IC to what was
            originally specified in the input file -- mostly used for running
            many conditions/parameter sets in series

        Returns
        -------
        self = an updated ODE class
        """

        if self.solver == 'vode':
            if not reset:
                y0 = vode_y(t0, self.network)
                self.y0 = y0
            else:
                y0 = self.y0
            self.ode_obj.set_initial_value(y0, t0)

    def set_user_params(self):
        """
        This function associates the network model with the residual/right hand
        side function employed by the solver.

        Parameters
        ----------
        network = a network object containing all the additional data

        Returns
        -------
        self = an updated ODE class
        """

        if self.solver == 'vode':
            self.ode_obj.set_f_params(self.network)

    def integrate(self, t_start, t_stop):
        """
        This function will take a time step towards steady state with the current
        specified network.

        Parameters
        ----------
        network = the current network object
        t_start = the starting time for integration (s)
        t_stop = the stopping time for integration (s)

        Returns
        -------
        network = the updated network object
        """

        # Make a call to the integrator
        if self.solver == 'vode':
            self.ode_obj.integrate(t_stop)
