""" This is a collection of classes and functions for modeling networks of ideal
reactors. It draws significant inspiration from graph theory as a network of
reactors is, in essence, a directed graph with nodes (sources, sinks, junctions,
and reactors) and edges (streams of flowing material connecting the nodes). """

from math import pi
from pyropy import util, gas
import numpy as np

class Node():
    """
    The base class for any of the basic control volume nodes in the network. We
    have four types of nodes that extend this base class: sources, sinks,
    junctions, and reactors. Sources, sinks, and junctions are for connecting
    reactors and only have information about the flow rates in to or out of
    other nodes. Sources (sinks) have only outflows (inflows), while junctions
    have both. Reactors are nodes that have the ability to have chemistry and
    mass inventory. Reactors are also permitted to convert streams of one phase
    into another (e.g., for a gasification reaction converting solids to gas) or
    to have an arbitrary number of streams (for a batch/semi-batch reactor).
    """

    # Keep a list of the maximum node_id number already used so that when we add
    # a new node, we can just increment to the next one. This doesn't really
    # facilitate deleting nodes (I suppose we can just delete all their
    # streams), but I don't really anticipate needing to adjust the network on
    # the fly. If we need to delete nodes, we should delete all streams
    # connecting to the deleted node(s) before deleting the nodes. Then we can
    # update all of the node ID numbers, starting at the end of the list and
    # working down. (I think.) I'll leave node deletion for later if we actually
    # need it.
    max_node_id = 0 # Class variable, shared by all objects of this class

    def __init__(self):
        raise NotImplementedError(
            'No initialization method implemented for generic Node objects.')

    def add_stream(self, stream):
        """
        Parameters
        ----------
        stream = a stream to add to the node

        Returns
        -------
        self = an updated Node object with an additional stream
        """

        # Check if the proposed stream has a terminus on the node
        if stream.sink != self.node_id and stream.source != self.node_id:
            raise ValueError('Attempted to add a new stream to node ' +
                str(self.node_id) + ' that does not begin or end at the node.')

        # OK to add the stream
        if stream.sink == self.node_id:
            self.streams_in.append(stream.stream_id)
        if stream.source == self.node_id:
            self.streams_out.append(stream.stream_id)

    def check(self):
        """
        This is a stub function for checking basic nodes. It doesn't do
        anything.
        """
        raise NotImplementedError(
            'No check function implemented for generic Node objects')

class Source(Node):
    """
    This class implements a source node. It differs from the base class mostly
    in that it provides facilities for adding outbound streams but not inbound
    streams. It also permits only a single stream per node.
    """

    def __init__(self, pos=None, streams=None, name=None):
        """
        Parameters
        ----------
        pos = x, y, z Cartesian coordinates (m) of the node center
        streams = a list of stream objects connecting this node to its
            neighbors (optional)
        name = the name of the node (string), useful for user-friendly defining
            of network connections

        Returns
        -------
        self = a Node object
        """

        if pos is None:
            self.pos = [0.0, 0.0, 0.0] # Assume some default coordinates

        if streams is None:
            self.streams_in = [] # Empty list
            self.streams_out = [] # Empty list
        else:
            self.streams_out = [stream.stream_id for stream in streams if
                stream.source == self.node_id]

        self.node_id = Node.max_node_id
        Node.max_node_id += 1
        self.num_phases = 0

        if name is None:
            self.name = self.node_id
        else:
            self.name = name

    def check(self, streams):
        """
        This function checks a source node for validity (defined as having a
        single stream leaving it).

        Parameters
        ----------
        streams = a list of stream objects associated with the source

        Returns
        -------
        None, but throws a runtime error if the number of streams constraint is
            violated, the stream is not associated with the sink, or the stream
            does not begin at the source
        """

        # Check the number of streams
        if len(self.streams_out) > 1 or len(streams) > 1:
            raise ValueError('Too many streams attached to source node ' +
                str(self.node_id))

        if len(streams) > len(self.streams_out):
            raise ValueError('Some streams associated with source node ' +
                str(self.node_id) + ' flow into the node')

        # Check that the supplied stream objects have IDs that match the
        # expected stream IDs
        for stream in streams:
            if stream.stream_id not in self.streams_out:
                raise ValueError('Stream ' + str(stream.stream_id) +
                    ' not associated with source ' + str(self.node_id))

        # Check that the supplied stream object begins at the source -- if we
        # get here, we know we have only a single stream object
        if streams[0].source != self.node_id:
            raise ValueError('Stream ' + str(streams[0].stream_id) +
                ' does not begin at source ' + str(self.node_id))

class Sink(Node):
    """
    This class implements a sink node. It differs from the base class mostly
    in that it provides facilities for adding inbound streams but not outbound
    streams.
    """

    def __init__(self, pos=None, streams=None, name=None):
        """
        Parameters
        ----------
        pos = x, y, z Cartesian coordinates (m) of the node center
        streams = a list of stream objects connecting this node to its
            neighbors (optional)
        name = the name of the node (string), useful for user-friendly defining
            of network connections

        Returns
        -------
        self = a Node object
        """

        if pos is None:
            self.pos = [0.0, 0.0, 0.0] # Assume some default coordinates

        if streams is None:
            self.streams_in = [] # Empty list
            self.streams_out = [] # Empty list
        else:
            self.streams_in = [stream.stream_id for stream in streams if
                stream.sink == self.node_id]

        self.node_id = Node.max_node_id
        Node.max_node_id += 1
        self.num_phases = 0

        if name is None:
            self.name = self.node_id
        else:
            self.name = name

    def check(self, streams):
        """
        This function checks a sink node for validity (defined as having a
        single stream entering it).

        Parameters
        ----------
        streams = a list of stream objects associated with the source

        Returns
        -------
        None, but throws a runtime error if the number of streams constraint is
            violated, the stream is not associated with the sink, or the stream
            does not end at the sink
        """

        # Check the number of streams
        if len(self.streams_in) > 1 or len(streams) > 1:
            raise ValueError('Too many streams attached to sink node ' +
                str(self.node_id))

        if len(streams) > len(self.streams_in):
            raise ValueError('Some streams associated with sink node ' +
                str(self.node_id) + ' flow out of the node')

        # Check that the supplied stream objects have IDs that match the
        # expected stream IDs
        for stream in streams:
            if stream.stream_id not in self.streams_in:
                raise ValueError('Stream ' + str(stream.stream_id) +
                    ' not associated with sink ' + str(self.node_id))

        # Check that the supplied stream object ends at the sink -- if we
        # get here, we know we have only a single stream object
        if streams[0].sink != self.node_id:
            raise ValueError('Stream ' + str(streams[0].stream_id) +
                ' does not end at sink ' + str(self.node_id))

class Junction(Node):
    """
    This class implements a general junction. A junction is a node that is
    permitted to have an arbitrary number of entering and exiting streams, all
    of a single phase. There must be at least one entering stream and one
    exiting stream. After initialization, additional streams may be added as
    desired.
    """

    def __init__(self, pos=None, streams=None, name=None):
        """
        Parameters
        ----------
        pos = x, y, z Cartesian coordinates (m) of the node center
        streams = a list of stream objects connecting this node to its
            neighbors (optional)
        name = the name of the node (string), useful for user-friendly defining
            of network connections

        Returns
        -------
        self = a Node object
        """

        if pos is None:
            self.pos = [0.0, 0.0, 0.0] # Assume some default coordinates

        if streams is None:
            self.streams_in = [] # Empty list
            self.streams_out = [] # Empty list
        else:
            self.streams_in = [stream.stream_id for stream in streams if
                stream.sink == self.node_id]
            self.streams_out = [stream.stream_id for stream in streams if
                stream.source == self.node_id]

        self.node_id = Node.max_node_id
        Node.max_node_id += 1
        self.num_phases = 0

        if name is None:
            self.name = self.node_id
        else:
            self.name = name

    def check(self, streams):
        """
        Parameters
        ----------
        None

        Returns
        -------
        None, but throws a runtime error if too few streams are attached to the
            junction, they do not start/end at the junction, there are only
            inflows or outflows, or the streams are not all the same phase
        """

        # Check that we have enough streams for a junction
        if len(self.streams_in) + len(self.streams_out) < 2:
            raise ValueError('Too few streams specified for junction node' +
                str(self.node_id))
        elif len(self.streams_in) < 1 or len(self.streams_out) < 1:
            raise ValueError('Junction ' + str(self.node_id) +
                'has only inflows or outflows, but needs at least one of each.')

        # Check that each stream is associated with the junction and that each
        # stream has a terminus (start or end) at the junction. We also count
        # the number of streams flowing in and out
        num_in_streams = 0
        num_out_streams = 0
        for stream in streams:
            if stream.stream_id not in (self.streams_in + self.streams_out):
                raise ValueError('Stream ' + str(stream.stream_id) +
                    ' not associated with junction ' + str(self.node_id))

            if stream.sink == self.node_id:
                num_in_streams += 1
            elif stream.source == self.node_id:
                num_out_streams += 1
            else:
                raise ValueError('Stream ' + str(stream.stream_id) +
                    ' associated with junction ' + str(self.node_id) +
                    'does not start/end at the junction.')

        # If we do not have both inflows and outflows, we have a problem
        if num_in_streams == 0 or num_out_streams == 0:
            raise ValueError('Junction ' + str(self.node_id) +
                'has only inflows or outflows, but needs at least one of each.')

        # Check that each stream has the same phase as the first
        phase = streams[0].phase
        for stream in streams:
            if stream.phase != phase:
                raise ValueError('Not all streams asociated with junction ' +
                    str(self.node_id) + ' are of the same phase.')

class Reactor(Node):
    """
    This class implements a reactor node. Reactor nodes may have any number or
    type of connecting streams and also have mass inventory. They may have any
    number of phases.

    They may also have a kinetic scheme that returns the species production
    rates.
    """

    def __init__(self, diam, height, pos=None, streams=None, name=None):
        """
        Parameters
        ----------
        diam = reactor diameter, assuming cylindrical reactor (m)
        height = reactor height (m)
        pos = x, y, z position in Cartesian coordinates
        streams = any connecting stream objects

        Returns
        -------
        self = a reactor object
        """

        # Basic attributes
        if pos is None:
            self.pos = [0.0, 0.0, 0.0] # Assume some default coordinates

        if streams is None:
            self.streams_in = [] # Empty list
            self.streams_out = [] # Empty list
        else:
            self.streams_in = [stream.stream_id for stream in streams if
                stream.sink == self.node_id]
            self.streams_out = [stream.stream_id for stream in streams if
                stream.source == self.node_id]

        self.node_id = Node.max_node_id
        Node.max_node_id += 1
        self.num_phases = 0

        if name is None:
            self.name = self.node_id
        else:
            self.name = name

        # Update the reactor with its geometric information
        self.diam = float(diam)
        self.height = float(height)
        self.vol = pi * diam**2 * height / 4.0

        # Initialize empty lists for all of the other parameters
        self.species = []
        self.mass_fracs = []
        self.masses = []
        self.mw = []
        self.dp = []
        self.rho = []
        self.rho_avg = []
        self.eps = []
        self.mu = []
        self.phase_type = []
        self.cp = []

    def set_TP(self, T, P):
        """
        This function sets the temperature and pressure in the reactor.

        Parameters
        ----------
        T = temperature (K)
        P = absolute pressure (kPa)

        Returns
        -------
        self = an updated reactor object with T, P set
        """

        self.T = T
        self.P = P

    def set_energy_balance_params(self, balance_type=None, U=None, Text=None,
        Asurf=None, cp=None):
        """
        This function is responsible for setting the parameters needed for the
        energy balance for this reactor.

        Parameters
        ----------
        balance_type = string setting the type of energy balance to solve:
            'isothermal': isothermal (default)
            'adiabatic': adiabatic
            'Ucoeff': heat transfer coefficient with surface area
        U = heat transfer coefficient for reactor (only used if 'Ucoeff' is
            balance type) (W/m^2/K)
        Text = external temperature (K) for non-adiabatic non-isothermal
            reactors
        Asurf = external surface area (m^2) for heat transfer (only applies to
            'Ucoeff' balance type); defaults to external geometric surface area
            if not specified and 'Ucoeff' used
        cp = array of heat capacities for each phase (J/kg/K)

        Returns
        -------
        self = an updated reactor object with the appropriate heat transfer
            characteristics
        """

        if balance_type is None:
            self.Q_balance_type = 'isothermal'
        elif balance_type not in ['isothermal', 'adiabatic', 'Ucoeff']:
            raise UserWarning('Invalid energy balance type ' + str(balance_type)
                + ' for reactor ' + str(self.node_id) +
                '. Using isothermal balance instead.')
            self.Q_balance_type = 'isothermal'
        else:
            self.Q_balance_type = balance_type

        if self.Q_balance_type != 'isothermal' and cp is None:
            raise ValueError('Energy balance solved for reactor ' +
                str(self.node_id) +
                ' but no vector of heat capacities specified.')
        else:
            self.cp = cp

        if self.Q_balance_type == 'Ucoeff':
            if U is None or Text is None:
                raise ValueError('General energy balance requested for reactor '
                    + str(self.node_id) + ' but overall heat transfer ' +
                    'coefficient and/or external temperature not specified.')
            else:
                self.Ucoeff = U
                self.Text = Text
            if Asurf is None:
                raise UserWarning('No heat transfer area specified for reactor '
                    + str(self.node_id) + ' in energy balance. ' +
                    'Using external geometric surface area.')
                self.Asurf = pi * self.diam * self.height
            else:
                self.Asurf = Asurf

    def add_phase(self, species=None, mass_fracs=None, mw=None,
            phase=None, mu=None, eps=None, eos=None, dp=None, rho=None):
        """
        Wrapper to add a generic phase to the reactor. Based on the values
        supplied, the appropriate phase type is added.

        Parameters
        ----------
        species = an array of species names
        mass_fracs = an array of mass fractions
        mw = an array of molar masses (g/mol)
        phase = the ID number assigned to the phase (None specifies gas)
        mu = the viscosity of the mixture (kg/m-s) for liquids/gases
        eps = the volume fraction occupied by the phase
        eos = an equation of state function for the vapor phase for calculating
            the density (kg/m^3); None will use the ideal gas law (built-in)
        dp = an array of solids particle diameters for each species (m)
        rho = array of mass densities for solids/liquids (kg/m^3) or None for
            gases (in this case the density comes from the equation of state)
        """

        if species is None:
            raise ValueError('No species names specified in phase description.')
        if mass_fracs is None:
            raise ValueError('No mass fractions specified in phase description.')
        if mw is None:
            raise ValueError('No molecular weights specified in phase description.')

        if (phase is None) or (phase == 0): # gas phase
            if mu is None:
                raise ValueError('No viscosity specified for gas phase.')
            self._add_gas_phase(species, mass_fracs, mw, mu, eps=eps, eos=eos)
        elif (dp is not None) and (rho is not None): # solids phase
            self._add_solids_phase(phase, species, mass_fracs, mw, dp, rho, eps)
        elif (mu is not None) and (rho is not None): # liquid phase
            self._add_liquid_phase(phase, species, mass_fracs, mw, mu, rho, eps,
                eos=eos)
        else:
            raise ValueError('Inconsistent phase parameters specified')

    def _add_gas_phase(self, species, mass_fracs, mw, mu, eps=None, eos=None):
        """
        Adds a gas phase to the reactor. Typically this will be the first phase
        added. The density is specified by the ideal gas law by default,
        although some other equation of state may be optionally supplied by tge
        user.

        Parameters
        ----------
        species = a list of the gas-phase species in the reactor
        mass_fracs = an array of the gas-phase mass fractions
        mw = an array of the gas phase molar masses (g/mol)
        mu = the viscosity of the mixture (kg/m/s)
        eps = the volume fraction of the gas phase; if not supplied, it will be
            calculated from the sum of the other phases' volume fractions
        eos = an equation of state function for calculating the density of the
            gas (kg/m^3); if supplied it should be a function of T, P, the mass
            fractions, and the molar masses and return the density (defaults to
            the ideal gas law)

        Returns
        -------
        self = an updated reactor node with a gas phase
        """

        # Extract the temperature and pressure
        T = self.T
        P = self.P

        # Set the number of phases
        self.num_phases += 1

        # Insert the information about this gas phase as the first entry. We
        # treat the bubble and emulsion phases as separate reactors.
        self.species.insert(0, species)
        self.mu.insert(0, mu)
        self.mass_fracs.insert(0, mass_fracs)
        self.mw.insert(0, mw)
        if eps is not None:
            self.eps.insert(0, eps) # Volume fraction of gas
        else:
            eps_total = sum(self.eps)
            self.eps.insert(0, 1.0 - eps_total)
        self.phase_type.insert(0, 'g') # Gas phase
        if eos is None: # Ideal gas law from the library
            mwa = util.mw_avg(mw, mass_fracs=mass_fracs)
            rho = gas.rhog(mwa, P, T)
        else: # User-supplied EOS
            try:
                rho = eos(T, P, mass_fracs, mw) # Must be user-supplied
            except: # Something went wrong, default to ideal gas law
                raise UserWarning(
                    'Invalid equation of state calculation for gas density in reactor '
                    + str(self.node_id) + '. Using ideal gas law instead.')
                mwa = util.mw_avg(mw, mass_fracs=mass_fracs)
                rho = gas.rhog(mwa, P, T)
        self.rho.insert(0, rho)
        self.rho_avg.insert(0, rho)
        mass = self.rho_avg[0] * self.vol * self.eps[0]
        self.masses.insert(0, mass)
        self.dp.insert(0, None)

    def _add_solids_phase(self, phase, species, mass_fracs, mw, dp, rho, eps):
        """
        Adds a solids phase to the reactor. There may be an arbitrary number of
        solids phases, but typically we will only need one.

        Parameters
        ----------
        phase = the index of the phase to add to this reactor
        species = a list of names of the solid species
        mass_fracs = an array of mass fractions for each of the components
            in the solids phase
        mw = array of molar masses (g/mol) of the solids species
        dp = array of particle diameters (m) for the solids species
            corresponding to the mass fractions
        rho = array of mass densities (kg/m^3) for the solids species in this
            phase
        eps = volume fraction occupied by this solids phase

        Returns
        -------
        self = an updated reactor with a new solids phase
        """

        # Find the current number of phases
        num_phases = self.num_phases

        # Calculate the average density and mass
        rho_avg = util.rho_avg(rho, mass_fracs)
        mass = rho_avg * self.vol * eps

        # The new phase has an index larger than the allocated structure
        if phase >= num_phases:
            # Add empty entries, if needed, to ensure that we place the new
            # phase in the correct position
            for i in range(num_phases, phase):
                self.species.append(None)
                self.eps.append(0.0)
                self.mass_fracs.append(None)
                self.masses.append(None)
                self.mw.append(None)
                self.dp.append(None)
                self.rho.append(None)
                self.rho_avg.append(None)
                self.mu.append(None)
                self.phase_type.append(None)

            # Add the solids properties to the object
            self.species.append(species)
            self.eps.append(eps)
            self.mass_fracs.append(mass_fracs)
            self.mw.append(mw)
            self.dp.append(dp)
            self.rho.append(rho)
            self.rho_avg.append(rho_avg)
            self.masses.append(mass)
            self.mu.append(None)
            self.phase_type.append('s')

        # We already have a slot for this phase, so fill it in
        else:

            # Add the solids properties to the object
            self.species[phase] = species
            self.eps[phase] = eps
            self.mass_fracs[phase] = mass_fracs
            self.mw[phase] = mw
            self.dp[phase] = dp
            self.rho[phase] = rho
            rho_avg = util.rho_avg(rho, mass_fracs)
            self.rho_avg[phase] = rho_avg
            self.masses[phase] = mass
            self.mu[phase] = None
            self.phase_type[phase] = 's'

        # Update the gas phase volume fraction and check for volume fraction
        # summing to 1
        self.eps[0] = 1 - sum(self.eps[1:])
        if self.eps[0] < 0:
            raise ValueError('Negative gas phase volume fraction detected when'
            + ' adding solids phase ' + str(self.num_phases)
            + ' to reactor node ' + str(self.node_id))

        # Update the number of phases in the reactor
        self.num_phases = len(self.eps)

    def _add_liquid_phase(self, phase, species, mass_fracs, mw, mu, rho, eps,
        eos=None):
        """
        Adds a liquid phase to the reactor. There may be an arbitrary number of
        liquid phases, but typically we will only need one.

        Parameters
        ----------
        phase = the index of the phase to add to this reactor
        species = a list of names of the solid species
        mass_fracs = an array of mass fractions for each of the components in
            the liquid phase
        mw = array of molar masses (g/mol) of the liquid species
        mu = vicosity of the liquid mixture (kg/m/s)
        rho = array of mass densities (kg/m^3) for the liquid species in this
            phase
        eps = volume fraction occupied by this liquid phase
        eos = an equation of state function for calculating the density of the
            gas (kg/m^3); if supplied it should be a function of T, P, the mass
            fractions, and the molar masses and return the density (defaults to
            constant density based on pure component densities if not supplied)

        Returns
        -------
        self = an updated reactor with a new liquid phase
        """

        # Find the current number of phases
        num_phases = self.num_phases

        # Calculate the average density and mass
        if eos is None: # Ideal gas law from the library
            rho_avg = util.rho_avg(rho, mass_fracs)
        else: # User-supplied EOS
            rho_avg = eos(T, P, mass_fracs, mw) # Must be user-supplied
        mass = rho_avg * self.vol * eps

        # The new phase has an index larger than the allocated structure
        if phase >= num_phases:
            # Add empty entries, if needed, to ensure that we place the new
            # phase in the correct position
            for i in range(num_phases, phase):
                self.species.append(None)
                self.eps.append(0.0)
                self.mass_fracs.append(None)
                self.masses.append(None)
                self.mw.append(None)
                self.dp.append(None)
                self.rho.append(None)
                self.rho_avg.append(None)
                self.mu.append(None)
                self.phase_type.append(None)

            # Add the liquid properties to the object
            self.species.append(species)
            self.eps.append(eps)
            self.mass_fracs.append(mass_fracs)
            self.mw.append(mw)
            self.dp.append(dp)
            self.rho.append(rho)
            self.rho_avg.append(rho_avg)
            self.masses.append(mass)
            self.mu.append(None)
            self.phase_type.append('l')

        # We already have a slot for this phase, so fill it in
        else:

            # Add the solids properties to the object
            self.species[phase] = species
            self.eps[phase] = eps
            self.mass_fracs[phase] = mass_fracs
            self.mw[phase] = mw
            self.dp[phase] = dp
            self.rho[phase] = rho
            rho_avg = util.rho_avg(rho, mass_fracs)
            self.rho_avg[phase] = rho_avg
            self.masses[phase] = mass
            self.mu[phase] = mu
            self.phase_type[phase] = 'l'

        # Update the gas phase volume fraction and check for volume fraction
        # summing to 1
        self.eps[0] = 1 - sum(self.eps[1:])
        if self.eps[0] < 0:
            raise ValueError('Negative gas phase volume fraction detected when'
            + ' adding liquid phase ' + str(self.num_phases)
            + ' to reactor node ' + str(self.node_id))

        # Update the number of phases in the reactor
        self.num_phases = len(self.eps)

    def add_kinetics(self, rate_law, rtd=None):
        """
        This function adds a kinetic scheme (rate law and residence time
        distributions) to the reactor.

        Parameters
        ----------
        rate_law = a user-defined function that encodes the kinetic scheme in a
            form that converts the conditions in the reactor to species rates of
            change
        rtd = a list of residence time distributions (one per phase) associated
            with the kinetic scheme (optional)

        Returns
        -------
        self = an updated reactor with a kinetic scheme
        """

        self.rate_law = rate_law

        if rtd is not None and self.num_phases != len(rtd):
            raise ValueError('Number of phases in reactor ' + str(self.node_id)
                + ' does not match the number of supplied residence time distributions.')
        self.rtd = rtd

    def get_species_rates(self):
        """
        This function calculates the species production rates with the embedded
        rate law. It is a convenience interface to the rate law function.

        Parameters
        ----------
        None passed. Uses specified reactor conditions.

        Returns
        -------
        self = an updated reactor with the species and enthalpy production rates
            specified
        """

        if self.rate_law is None:
            # No kinetics -- all species rates of change in all phases are zero
            # and there is no enthalpy change due to reaction
            rates = [np.zeros(len(mass_fracs[p])) for p in
                range(self.num_phases)]
            dH_rxn = 0.0
        else:
            # Pass all of the parameters characterizing this reactor to the
            # rate law. The actual rate law function has to be supplied by the
            # user and can be extremely general. It should include all of the
            # kinetics, stoichiometry, enthalpy changes, etc.
            T = self.T
            P = self.P
            mw = self.mw
            masses = self.masses
            mass_fracs = self.mass_fracs
            eps = self.eps
            dp = self.dp
            mu = self.mu
            vol = self.vol
            rates, dH_rxn = self.rate_law(T, P, vol, mw, masses, mass_fracs,
                eps, dp, mu)

        self.rates = rates # Arrays of species rates (kg/s) for each phase
        self.phase_rates = [sum(rates[p]) for p in range(self.num_phases)]
        self.dH_rxn = dH_rxn # Overall enthalpy change in reactor

    def get_phase_masses(self):
        """
        This function will calculate the total mass of each phase in the
        reactor.

        Parameters
        ----------
        None

        Returns
        -------
        self = an updated reactor object with an array of the masses of material
            in each phase
        """

        masses = []
        for i in range(self.num_phases):
            if self.phase_type[i] == 'g': # Gas
                mass = self.rho_avg[i] * self.vol * self.eps[i]
            else: # Solids and liquids
                # Get the mixture density, assuming additive volumes
                self.rho_avg[i] = util.rho_avg(self.rho[i], self.mass_fracs[i])
                mass = self.rho_avg[i] * self.vol * self.eps[i]
            masses.append(mass)

        self.masses = masses

    def check(self, streams):
        """
        Parameters
        ----------
        None

        Returns
        -------
        None, but throws a runtime error if there is a stream object thought to
            be associated with the reactor that isn't or that is not connected
            to it or if any of the connected streams have a phase or composition
            mis-match
        """

        # Check that each stream is associated with the reactor and that each
        # stream has a terminus (start or end) at the reactor. Also check that
        # each stream's phase has a corresponding phase in the reactor of the
        # same type and composition.
        for stream in streams:
            if stream.stream_id not in (self.streams_in + self.streams_out):
                raise ValueError('Stream ' + str(stream.stream_id) +
                    ' not associated with reactor ' + str(self.node_id))

            if stream.sink != self.node_id and stream.source != self.node_id:
                raise ValueError('Stream ' + str(stream.stream_id) +
                    ' associated with reactor ' + str(self.node_id) +
                    'does not start/end at the reactor.')

            if stream.phase > self.num_phases:
                raise ValueError('Phase number of stream ' +
                    str(stream.stream_id) +
                    ' is larger than the number of phases defined in reactor '
                    + str(self.node_id))

            if stream.phase_type != self.phase_type[stream.phase]:
                raise ValueError('Mismatch between phase of stream ' +
                    str(stream.stream_id) +
                    ' and the corresponding phase in reactor ' +
                    str(self.node_id))

            if stream.species != self.species[stream.phase]:
                raise ValueError('Mismatch between species of stream ' +
                    str(stream.stream_id) +
                    ' and the species in the corresponding phase in reactor ' +
                    str(self.node_id))

class Stream():
    """
    This class implements a flow stream for connecting nodes in the network.
    Each stream connects exactly two nodes in the network and contains all the
    information required for fully specifying that stream (i.e., the
    temperature, pressure, composition, phase number, flow rate, etc.). It is
    permissible to have two otherwise identical streams that are anti-parallel
    to each other (representing a closed flow loop between neighboring nodes).
    Each stream is assumed to have a single phase (gas, solids 1, 2, ..., etc.).
    """

    # Keep track of the maximum stream ID number
    max_stream_id = 0

    def __init__(self, name, source, sink):
        """
        Parameters
        ----------
        source = the name and ID of the source node (two element list)
        sink = the name and ID of the sink node (two element list)

        Returns
        -------
        self = a stream object
        """

        self.stream_id = Stream.max_stream_id
        Stream.max_stream_id += 1
        self.name = name
        self.source_name = source[0]
        self.source = source[1]
        self.sink_name = sink[0]
        self.sink = sink[1]

        # Make sure we don't have any self-looping streams
        if self.source == self.sink:
            raise ValueError('Stream ' + str(self.stream_id) +
                ' is a simple closed loop with identical source/sink IDs.')

    def init_conditions(self, T=None, P=None, species=None, mass_fracs=None,
        mw=None, vol_flow=None, phase=None, rho=None, mu=None, dp=None, H=None,
        eos=None):
        """
        This function initializes the flow conditions of the stream.

        Parameters
        ----------
        T = the temperature of the stream (K)
        P = the pressure of the stream (kPa)
        species = a list of species names in the stream
        mass_fracs = an array of the mass fractions of the species in the stream
        mw = an array of the molar masses (g/mol) of the species
        vol_flow = the volumetric flow rate (m^3/s)
        rho (optional) = array of species mass densities (kg/m^3) for solids and
            liquids (set to None for gases)
        mu (optional) = gas or liquid viscosity (kg/m/s)
        dp (optional) = array of particle diameters (m) for solids
        phase = the ID number of the phase, defaults to 0 (gas)
        H (optional) = array of specific enthalpies for each of the species
            (J/kg)
        eos (optional) = a function for calculating the density from an equation
            of state

        Returns
        -------
        self = a stream object with all of the physical conditions set
        """

        if T is None:
            raise ValueError(
                'No temperature specified during stream initialization')
        if P is None:
            raise ValueError(
                'No pressure specified during stream initialization')
        if species is None:
            raise ValueError(
                'No species names specified during stream initialization')
        if mass_fracs is None:
            raise ValueError(
                'No mass fractions specified during stream initialization')
        if mw is None:
            raise ValueError(
                'No molecular weights specified during stream initialization')
        if vol_flow is None:
            raise ValueError(
                'No volumetric flow rate specified during stream initialization')

        # Common properties
        self.T = float(T)
        self.P = float(P)
        self.species = species
        self.mass_fracs = mass_fracs
        self.mw = mw
        self.vol_flow = vol_flow

        # Enthalpy
        if H is None:
            self.H = np.zeros(len(mass_fracs))
        else:
            self.H = H

        # Phase number
        if phase is None:
            self.phase = 0
        else:
            self.phase = phase

        # Phase-type-dependent properties
        if rho is None and mu is not None:
            # Gas
            mwa = util.mw_avg(mw, mass_fracs=mass_fracs)
            self.rhop = None
            self.mu = mu
            self.dp = None
            self.phase_type = 'g'
        elif rho is not None and mu is not None:
            # Liquid
            self.rhop = rho
            self.mu = mu
            self.dp = None
            self.phase_type = 'l'
        elif rho is not None and mu is None and dp is not None:
            # Solid
            self.rhop = rho
            self.mu = None
            self.dp = dp
            self.phase_type = 's'
        else:
            # Erroneous phase
            raise ValueError(
                'Incomplete specification of phase properties for stream ' +
                str(self.stream_id) + ' and phase ' + str(phase))

        # Set the density
        self.find_rho(eos=eos)

        # Set the mass flow rate (volumetric rate multiplied by density)
        self.find_mass_flow()

    def find_rho(self, eos=None):
        """
        This function will calculate the mass density of the stream.

        Parameters
        ----------
        None

        Returns
        -------
        self = an update stream object with the correct density
        """

        if eos is not None: # User-defined equation of state
            self.rho_avg = eos(self.T, self.P, self.mass_fracs, self.mw)
        elif self.phase_type == 'g': # Use ideal gas
            mwa = util.mw_avg(self.mw, mass_fracs=self.mass_fracs)
            self.rho_avg = gas.rhog(mwa, self.P, self.T)
        else: # Solids and liquids assume ideal solution and additive volumes
            self.rho_avg = util.rho_avg(self.rhop, self.mass_fracs)

    def find_mass_flow(self):
        """
        This function will calculate the mass flow rate of the stream.

        Parameters
        ----------
        None

        Returns
        -------
        self = an updated stream object with the proper mass flow rate
        """
        self.mass_flow = self.rho_avg * self.vol_flow

    def find_vol_flow(self):
        """
        This function will calculate the volumetric flow rate of the stream.

        Parameters
        ----------
        None

        Returns
        -------
        self = an updated stream object with the proper volumetric mass flow
            rate
        """

        self.vol_flow = float(self.mass_flow) / self.rho_avg

class Network():
    """
    This class bundles all of the nodes and streams together into a single
    Network object along with the user-defined solution algorithm. This
    facilitates interfacing the reactor network with a larger optimization or
    sensitivity analysis/uncertainty quantification code.
    """

    def __init__(self, nodes, streams):
        """
        Initializes the network topology. Error checking should be done after
        the initial conditions are specified.

        Parameters
        ----------
        nodes = a list of nodes in the network
        streams = a list of streams in the network

        Returns
        -------
        self = a Python object bundling the nodes and streams together with the
            solution algorithm
        """

        # Assign the node and stream lists
        self.nodes = nodes
        self.streams = streams

        # Map the node and stream lists to ID numbers
        self.map_nodes_and_streams()

        # Count the number of reactors in the network, excluding the sinks,
        # sources, and junctions
        self.num_reactors = sum([isinstance(node, Reactor) for node in self.nodes])

        # Create a list of the number of species in each phase
        self.num_phases = [len(node.mass_fracs) for node in self.nodes if
            isinstance(node, Reactor)]
        self.num_species_per_phase = np.zeros((max(self.num_phases),
            self.num_reactors), dtype=np.int)
        rxtr_idx = 0
        for node in self.nodes:
            if isinstance(node, Reactor):
                num_species = np.array([len(mass_frac_array) for mass_frac_array
                    in node.mass_fracs], dtype=np.int)
                for phase_idx in range(len(num_species)):
                    self.num_species_per_phase[phase_idx, rxtr_idx] = num_species[phase_idx]
                rxtr_idx += 1

        # Count the number of streams in the network
        self.num_streams = len(streams)

        # Iterate over all the streams, assigning each one to its associated
        # nodes
        for stream in streams:
            source_node_idx = self.node_map[stream.source]
            self.nodes[source_node_idx].add_stream(stream)
            sink_node_idx = self.node_map[stream.sink]
            self.nodes[sink_node_idx].add_stream(stream)

    def map_nodes_and_streams(self):
        """
        This function iterates over all the nodes and streams in the network and
        creates maps of which node and stream ID numbers correspond to which
        node and stream list indices. Unless nodes or streams have been deleted,
        the list indices and the object ID numbers are equivalent. It also
        creates a dictionary mapping node names to node IDs.

        Parameters
        ----------
        None

        Returns
        -------
        None, but constructs the maps and stores them in the network object.
        """

        self.node_name_map = {node.name: node.node_id for node in self.nodes}
        self.node_map = {self.nodes[i].node_id: i for i in
            range(len(self.nodes))}
        self.stream_name_map = {stream.name: stream.stream_id for stream in self.streams}
        self.stream_map = {self.streams[i].stream_id: i for i in
            range(len(self.streams))}

    def get_node_idx(self, node_name):
        """
        This function will get the location of the node in the node list given
        its user-defined name.

        Parameters
        ----------
        node_name = the name of the node

        Returns
        -------
        node_idx = the index of the node in the list of nodes
        """

        node_id = self.node_name_map[node_name]
        node_idx = self.node_map[node_id]
        return node_idx

    def get_stream_idx(self, stream_name):
        """
        This function will get the location of the stream in the stream list
        given its user-defined name.

        Parameters
        ----------
        stream_name = the name of the stream

        Returns
        -------
        stream_idx = the index of the stream in the list of streams
        """

        stream_id = self.stream_name_map[stream_name]
        stream_idx = self.stream_map[stream_id]
        return stream_idx

    def rev_stream(self, stream_id):
        """
        This function will swap the source and sink nodes to reverse the flow of
        the stream.

        Parameters
        ----------
        stream_id = the ID number of the stream whose flow is to be reversed

        Returns
        -------
        self = an updated network
        """

        # Find the nodes and the stream in the lists of node/stream objects
        stream_idx = self.stream_map[stream_id] # Index to stream object in list
        source_name = self.streams[stream_idx].source_name # name of source node
        source_id = self.streams[stream_idx].source # ID of source node
        sink_name = self.streams[stream_idx].sink_name # name of sink node
        sink_id = self.streams[stream_idx].sink     # ID of sink node
        source_idx = self.node_map[source_id]  # Index of source node in list
        sink_idx = self.node_map[sink_id]      # Index of sink node in list

        # Update the nodes
        # Source node to sink node
        self.nodes[source_idx].streams_out.remove(stream_id)
        self.nodes[source_idx].streams_in.append(stream_id)

        # Sink node to source node
        self.nodes[sink_idx].streams_in.remove(stream_id)
        self.nodes[sink_idx].streams_out.append(stream_id)

        # Update the stream source and sink
        self.streams[stream_idx].source_name = sink_name
        self.streams[stream_idx].source = sink_id
        self.streams[stream_idx].sink_name = sink_name
        self.streams[stream_idx].sink = source_id

    def check(self):
        """
        This function will check all nodes and streams to ensure
        self-consistency in the connectivity of the network.

        Parameters
        ----------
        None

        Returns
        -------
        None, but throws runtime errors if an error in the network graph is
            found.
        """

        # Iterate over all nodes, checking:
        #   1.  Sources/sinks
        #       a.  Only have a single stream
        #       b.  Have the appropriate terminus in the associated stream
        #   2.  Junctions
        #       a.  Have at least two streams
        #       b.  At least one stream enters and at least one exits
        #       c.  All streams have the same phase number
        #   3.  Reactors
        #       a.  All streams associated with the reactor have the reactor as
        #           a terminus
        #       b.  The phase numbers and species of all streams associated with
        #           the reactor are present in the reactor
        #   4.  All nodes are connected to the network if there are multiple
        #       nodes and any streams are present
        for node in self.nodes:

            # Extract the streams associated with this node
            node_streams = node.streams_in + node.streams_out
            stream_indices = [self.stream_map[i] for i in node_streams]
            streams = [self.streams[i] for i in stream_indices]

            # Checks the node according to the constraints placed on it
            node.check(streams)

            # Now check if this node is not connected to anything and we have
            # one or more streams specified and more than one reactor (if we
            # have a single, isolated batch reactor, then this is a valid
            # 'network')
            if (len(self.nodes) > 1 and len(self.streams) > 0 and
                len(node.streams_in) == 0 and len(node.streams_out) == 0):
                raise ValueError('Node ' + str(node.node_id) +
                    ' is not connected to the larger network.')

        # Iterate over all streams, checking that all source/sink points for the
        # stream exist in the list of nodes
        for stream in self.streams:
            source_id = stream.source
            sink_id = stream.sink
            if source_id not in self.node_map or sink_id not in self.node_map:
                raise ValueError('Stream ' + str(stream.stream_id) +
                    ' has a source/sink node that is not in the network.')

        # If we get this far, then we have verified that the network
        # connectivity is OK

    def get_mass_change(self, reactor_id):
        """
        This function iterates over all of the streams entering and exiting the
        reactor and calculates the overall mass change in the reactor for each
        phase.

        Parameters
        ----------
        reactor_id = the node ID number for the reactor

        Returns
        -------
        mass_change = an array of mass accumulation (or depletion) rates (kg/s)
            for each phase in the specified reactor
        """

        # Get the node index
        node_idx = self.node_map[reactor_id]

        # Set up arrays with the inflow and outflow rates
        num_phases = self.nodes[node_idx].num_phases
        mass_in_rates = np.zeros(num_phases)
        mass_out_rates = mass_in_rates * 0.0

        # Calculate the mass flows in and out for each phase by iterating over
        # the associated streams
        for stream_in in self.nodes[node_idx].streams_in:
            stream_idx = self.stream_map[stream_in]
            phase_idx = self.streams[stream_idx].phase
            mass_in_rates[phase_idx] += self.streams[stream_idx].mass_flow
        for stream_out in self.nodes[node_idx].streams_out:
            stream_idx = self.stream_map[stream_out]
            phase_idx = self.streams[stream_idx].phase
            mass_out_rates[phase_idx] += self.streams[stream_idx].mass_flow

        # Get the net mass change due to all reactions; this should capture only
        # the interphase mass transfer (due to interfacial reactions and/or
        # phase phase exchange), as all homogeneous reactions should sum to zero
        source_rates = np.array(self.nodes[node_idx].phase_rates)

        # Calculate the net mass flow rates for all phases
        mass_change = mass_in_rates - mass_out_rates + source_rates
        return mass_change

    def get_species_mass_frac_change(self, reactor_id, phase_idx):
        """
        This function calculates the change in the mass fractions of the species
        in the specified reactor and phase.

        Parameters
        ----------
        reactor_id = ID number of the reactor
        phase_idx = index of the phase for which the mass fraction change is to
            be calculated

        Returns
        -------
        species_mass_frac_change = an array of rates of change for the mass
            fractions in the specified reactor and phase
        """

        # Get the node index
        node_idx = self.node_map[reactor_id]

        # Get the reactor information
        mass_frac_rxtr = np.asarray(self.nodes[node_idx].mass_fracs[phase_idx])
        mass_rxtr = self.nodes[node_idx].masses[phase_idx]
        species_rates = self.nodes[node_idx].rates[phase_idx]
        phase_rate = sum(species_rates)

        # Count the number of inlet streams of the proper phase
        num_in_streams = 0
        for stream_in in self.nodes[node_idx].streams_in:
            stream_idx = self.stream_map[stream_in]
            if phase_idx == self.streams[stream_idx].phase:
                num_in_streams += 1

        # Get the number of species
        num_species = len(self.nodes[node_idx].mass_fracs[phase_idx])

        # Extract the stream data and assign it to arrays
        mass_flow_in = np.zeros(num_in_streams)
        mass_frac_in = np.zeros((num_in_streams, num_species))
        s = 0
        for stream_in in self.nodes[node_idx].streams_in:
            stream_idx = self.stream_map[stream_in]
            if phase_idx == self.streams[stream_idx].phase:
                mass_flow_in[s] = self.streams[stream_idx].mass_flow
                mass_frac_in[s, :] = self.streams[stream_idx].mass_fracs
                s += 1

        # Evaluate the RHS of the species ODE. There are two sets of terms:
        #   1.  Flow terms involving the mass flow in and out of the reactor
        #   2.  Source terms involving the production of species due to
        #       reactions and phase change.

        # Flow terms
        flow_diff = np.dot(mass_flow_in, mass_frac_in - mass_frac_rxtr)

        # Source terms
        source_diff = species_rates - (mass_frac_rxtr * phase_rate)

        # Composite change
        mass_frac_change = (flow_diff + source_diff)/mass_rxtr

        return mass_frac_change

    def get_T_change(self, reactor_id):
        """
        This function calculates the temperature change in the reactor.

        Parameters
        ----------
        reactor_id = ID number of the reactor

        Returns
        -------
        dT = temperature change rate in the reactor (K/s)
        """

        # Get the node index
        node_idx = self.node_map[reactor_id]

        if self.nodes[node_idx].Q_balance_type == 'isothermal':
            # Isothermal, so we don't need to calculate anything
            dT = 0.0
            return dT
        else:
            # Non-isothermal, so we solve the energy balance

            # Extract the enthalpy change due to reaction (requires that the
            # reaction kinetics be specified first!)
            dH_rxn = self.nodes[node_idx].dH_rxn

            # Calculate the enthalpies of the connecting flow streams
            H_in_total = 0.0
            H_out_total = 0.0
            for stream_in in self.nodes[node_idx].streams_in:
                stream_idx = self.stream_map[stream_in]
                H_in = (np.dot(self.streams[stream_idx].mass_fracs,
                              self.streams[stream_idx].H)
                        * self.streams[stream_idx].mass_flow)
                H_in_total += H_in
            for stream_out in self.nodes[node_idx].streams_out:
                stream_idx = self.stream_map[stream_out]
                H_out = (np.dot(self.streams[stream_idx].mass_fracs,
                               self.streams[stream_idx].H)
                         * self.streams[stream_idx].mass_flow)
                H_out_total += H_out

            # Find the heat term, if present
            if self.nodes[node_idx].Q_balance_type == 'Ucoeff':
                # External heat transfer
                Q = (self.nodes[node_idx].U * self.nodes[node_idx].Asurf
                    * (self.nodes[node_idx].T - self.nodes[node_idx].Text))
            elif self.nodes[node_idx].Q_balance_type == 'adiabatic':
                Q = 0.0

            # Approximate overall energy balance -- NOTE: this assumes that the
            # mass in the reactor and the heat capacities of the phases are
            # *constant*. This is not quite right, but it makes the math and the
            # code a lot easier, and we are only interested in steady state
            # anyway. Primarily, it means we don't have to store the mass change
            # for each phase with the reactor and we don't have to explicitly
            # track the reference temperature used in calculating the enthalpy
            # of the material in the reactor.

            # Total enthalpy change
            H_change = H_in_total - H_out_total + dH_rxn + Q

            # Sum heat capacities and masses to get normalization factor
            cp_m = np.dot(self.nodes[node_idx].cp, self.nodes[node_idx].masses)

            # Temperature change
            dT = H_change / cp_m
            return dT

    def set_mass_flow_rates_func(self, mass_flow_func):
        """
        This function associates a user-defined function for calculating the
        mass flow rates of each stream with the reactor network. At the moment,
        this function must be written by hand by the user, although in the
        future it may be possible to automate it (e.g., with flowsheeting
        concepts) or to auto-generate it (e.g., from external data).

        Parameters
        ----------
        mass_flow_func = a *function* that will calculate the mass flow rates
            for each stream in the network

        Returns
        -------
        self = an updated network object with a function for iterating over the
            network nodes to calculate the mass flow rates for each stream
        """

        self._get_mass_flow_rates_func = mass_flow_func

    def get_mass_flow_rates(self, func=None):
        """
        This function is an interface for accessing the user-supplied function
        used to calculate the flow rates for each of the streams in the network.

        Parameters
        ----------
        func = a user-supplied function for calculating the flow rates

        Returns
        -------
        self = an updated network object with all of the stream flow rates
            specified
        """

        if func is not None:
            self._get_mass_flow_rates_func = func
        # NOTE: We need to explicitly pass self as this function is not bound to
        # the class.
        self._get_mass_flow_rates_func(self)

    def perturb_stream_properties(self, op_space_params):
        """
        This function is used to alter the properties of the inlet streams.

        Parameters
        ----------
        op_space_params = A dictionary with the stream properties that are to be
            perturbed. Keys are space delimited and have up to three fields.
            Field 1 is the stream name, field 2 is the type of property, and
            field three is the name of the species (used only if field 2 affects
            an individual species, such as the mass fraction).

        Returns
        -------
        self = an updated network object
        """

        # Modify any inlet stream parameters requested by the operating condition
        # scan
        vol_flow_streams = [] # List of stream indices with perturbed vol flows
        mass_flow_streams = [] # List of stream indices with perturbed mass flows
        rho_streams = [] # List of stream indices with perturbed densities
        if len(op_space_params) > 0: # Check for non-empty dict!
            for key in op_space_params:
                key_list = key.split()
                stream_name = key_list[0]
                stream_idx = self.get_stream_idx(stream_name)
                prop = key_list[1]
                prop_val = op_space_params[key]
                if prop in ['mw', 'mass_fracs', 'rho']: # Species property, changes density
                    if stream_idx not in rho_streams:
                        rho_streams.append(stream_idx)
                    spec_names = getattr(self.streams[stream_idx], 'species')
                    spec_name = key_list[2]
                    spec_idx = spec_names.index(spec_name)
                    prop_vec = getattr(self.streams[stream_idx], prop)
                    prop_vec[spec_idx] = prop_val
                    if prop == 'mass_fracs':
                        if prop_vec.size == 1:
                            raise UserWarning(
                                'Attempted to perturb the only mass fraction in a pure stream.')
                            prop_vec[0] = 1.0
                        else:
                            prop_vec[0] = 1.0 - sum(prop_vec[1:])
                    setattr(self.streams[stream_idx], prop, prop_vec)
                elif prop in ['dp', 'H']: # Species property, does not change density
                    spec_names = getattr(self.streams[stream_idx], 'species')
                    spec_name = key_list[2]
                    spec_idx = spec_names.index(spec_name)
                    prop_vec = getattr(self.streams[stream_idx], prop)
                    prop_vec[spec_idx] = prop_val
                    setattr(self.streams[stream_idx], prop, prop_vec)
                else: # Scalar property
                    if prop == 'vol_flow' and stream_idx not in vol_flow_streams:
                        vol_flow_streams.append(stream_idx)
                    elif prop == 'mass_flow' and stream_idx not in mass_flow_streams:
                        mass_flow_streams.append(stream_idx)
                    setattr(self.streams[stream_idx], prop, prop_val)

        # Recalculate the densities and volumetric and mass flows according to the
        # new parameters
        for stream_idx in rho_streams: # Recalculate density
            self.streams[stream_idx].find_rho()
        for stream_idx in vol_flow_streams: # Vol flow specified, find mass flow
            self.streams[stream_idx].find_mass_flow()
        for stream_idx in mass_flow_streams: # Mass flow specified, find vol. flow
            self.streams[stream_idx].find_vol_flow()

def create_nodes(nodes, rate_law):
    """
    This function is responsible for defining all of the nodes (sources, sinks,
    junctions, and reactors) in the network.

    Parameters
    ----------
    nodes = a dictionary keyed by name to dictionaries describing the nodes;
      each sub-dictionary has keys for the type, position, diameter, height,
      etc.
    rate_law = a function for calculating the reaction rates

    Returns
    -------
    nodes_list = a list of node-type objects in the network
    """

    # Initialize the list
    nodes_list = []

    for node_key in nodes:
        node = nodes[node_key]
        if node['type'] == 'source':
            nodes_list.append(Source(pos=node['pos'], name=node_key))
        elif node['type'] == 'sink':
            nodes_list.append(Sink(pos=node['pos'], name=node_key))
        elif node['type'] == 'junction':
            nodes_list.append(Junction(pos=node['pos'], name=node_key))
        elif node['type'] == 'reactor':
            nodes_list.append(Reactor(node['diam'], node['height'],
                pos=node['pos'], name=node_key))
            nodes_list[-1].set_TP(node['T'], node['P'])
            kwargs = node['Q_balance_params']
            nodes_list[-1].set_energy_balance_params(**kwargs)
            for phase in node['phases']:
                kwargs = node['phases'][phase]
                kwargs['phase'] = phase
                nodes_list[-1].add_phase(**kwargs)
            nodes_list[-1].add_kinetics(rate_law)

    # Create a dictionary pairing the names of the nodes to the node ID numbers
    node_name_map = {node.name: node.node_id for node in nodes_list}

    return nodes_list, node_name_map

def create_streams(streams, node_name_map):
    """
    This function is responsible for creating the streams that connect the nodes
    in the network. It does not specify the initial conditions in the streams.

    Parameters
    ----------
    streams = a dictionary with the stream names and the source and sink node
        names for each stream as a list
    node_name_map = a dictionary associating the names of the nodes in the
        network with their ID numbers

    Returns
    -------
    streams_list = a list of all the streams in the network
    """

    # Initialize the streams
    streams_list = []
    for stream_key in streams:
        stream = streams[stream_key]
        stream_pair = stream['nodes']
        source_list = [stream_pair[0], node_name_map[stream_pair[0]]]
        sink_list = [stream_pair[1], node_name_map[stream_pair[1]]]
        streams_list.append(Stream(stream_key, source_list, sink_list))
        streams_list[-1].init_conditions(**stream['properties'])

    # Create a dictionary pairing the names of the streams to the stream ID
    # numbers
    stream_name_map = {stream.name: stream.stream_id for stream in streams_list}

    return streams_list, stream_name_map

def setup_network(nodes_dict, streams_dict, rate_law, mass_flow_rates_func):
    """
    This function initializes the network object from the model input and
    returns a properly constructed Network object.

    Parameters
    ----------
    nodes_dict = a dictionary with the node parameters, as specified in
        model_input
    streams_dict = a dictionary with the stream parameters, as specified in
        model_input
    rate_law = a function to calculate the reaction rates that matches the call
        signature of the rate_law function of a Reactor node
    mass_flow_rates_func = a function to explicitly calculate the mass flow
        rates in the network

    Returns
    -------
    network = a Network object with all of the nodes and streams specified
    """

    # Create the nodes
    nodes_list, node_name_map = create_nodes(nodes_dict, rate_law)

    # Create the streams
    streams_list, stream_name_map = create_streams(streams_dict, node_name_map)

    # Create the Network object -- this also makes sure that all nodes have the
    # associated streams added to them
    network = Network(nodes_list, streams_list)

    # Check the network for connectivity and phase mismatch errors
    network.check()

    # Assign the function for getting the mass flow rates to the network
    network.set_mass_flow_rates_func(mass_flow_rates_func)

    # Finalize the mass flow rates for the internal and outlet streams
    network.get_mass_flow_rates()

    return network
