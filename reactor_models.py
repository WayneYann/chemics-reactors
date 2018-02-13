"""This module is a place to collect functions that can help with constructing
models of common reactor types. Specifically, the classes should be written to
return ordered dictionaries of the nodes, the streams, and a mass flow rates
function. The objects should be instantiated in model_input module, and any
initial and boundary condition properties should be specified there. The main
driver code will read the ordered dictionaries from model_input when
constructing the actual reactor_network object with all of its properties. A
few common reactor types are provided as examples. Others may be added as
needed."""

from collections import OrderedDict
import params

def batch(diam, height):
    """
    This function creates a simple ordered dictionary with a single reactor
    of the specified diameter and height.

    Parameters
    ----------
    diam = the reactor diameter (m)
    height = the reactor height (m)

    Returns
    -------
    nodes = an ordered dictionary with all of the basic node information
    streams = an ordered dictionary with all of the basic stream information
    """

    nodes = OrderedDict([('batch', 
                          {'type': 'reactor',
                           'pos': [0.0, 0.0, 0.0], # Not important for batch
                           'diam': diam,
                           'height': height})
                       ])
    streams = OrderedDict()
    return nodes, streams

def batch_mass_flow_rates(network):
    """
    This function calculates the mass flow rates between reactors in a batch
    network. There are no streams in a batch network, so it does nothing.

    Parameters
    ----------
    network = a network object; this is passed automatically when the mass
        flow rates function is assigned to the network object

    Returns
    -------
    None
    """

    pass # Minimum allowed function body to do nothing

def semibatch(diam, height):
    """
    This function creates a simple ordered dictionary with a single reactor
    of the specified diameter and height.

    Parameters
    ----------
    diam = the reactor diameter (m)
    height = the reactor height (m)

    Returns
    -------
    nodes = an ordered dictionary with all of the basic node information
    streams = an ordered dictionary with all of the basic stream information
    """

    nodes = OrderedDict([('gas_source',
                          {'type': 'source',
                           'pos': [0.0, 0.0, 0.0]}),
                         ('semibatch', 
                          {'type': 'reactor',
                           'pos': [0.0, 0.0, 0.0], # Not important for batch
                           'diam': diam,
                           'height': height})
                       ])
    streams = OrderedDict([('gas_inlet',
                                 {'nodes': ['gas_source', 'semibatch']})
                              ])
    return nodes, streams

def semibatch_mass_flow_rates(network):
    """
    This function calculates the mass flow rates between reactors in a
    network. There is a single stream in the semi-batch, so we only have
    sufficient code here to permit perturbations of the inlet parameters.

    Parameters
    ----------
    network = a network object; this is passed automatically when the mass
        flow rates function is assigned to the network object

    Returns
    -------
    network = an updated network object
    """

    # Perturb the inlet parameters
    op_space_params = getattr(params, 'op_space_params', None)
    if op_space_params is not None:
        network.perturb_stream_properties(op_space_params)

def cstr_train(diam, height, num_CSTRs=None, asPFR=None):
    """
    This function creates a simple ordered dictionary with a single reactor
    of the specified diameter and height.

    Parameters
    ----------
    diam = the reactor diameter (m)
    height = the reactor height (m)
    num_CSTRs = the number of CSTRs in the train; defaults to 1
    asPFR = a Boolean that determines whether the 'height' is the height of
        the entire reactor (True) or of each CSTR in the train (False);
        defaults to True

    Returns
    -------
    nodes = an ordered dictionary with all of the basic node information
    streams = an ordered dictionary with all of the basic stream information
    """

    # Initialize defaults of keyword arguments
    if num_CSTRs is None:
        nCSTRs = 1
    else:
        nCSTRs = num_CSTRs
    if asPFR is None:
        treat_as_PFR = True
    else:
        treat_as_PFR = asPFR

    if treat_as_PFR:
        cstr_height = height / nCSTRs
        train_height = height
    else:
        cstr_height = height
        train_height = height * nCSTRS

    # Source, sink nodes
    nodes = OrderedDict([('gas_source',
                          {'type': 'source',
                           'pos': [0.0, 0.0, 0.0]}),
                         ('gas_sink', 
                          {'type': 'sink',
                           'pos': [0.0, 0.0, train_height]})
                       ])
    # Reactor nodes
    pos = 0.0
    for nCSTR in range(nCSTRs):
        cstr_name = 'R' + str(nCSTR)
        nodes[cstr_name] = {'type': 'reactor',
                            'pos': [0.0, 0.0, pos],
                            'diam': bed_diam,
                            'height': cstr_height}
        pos += cstr_height

    # Inlet, outlet streams
    streams = OrderedDict([('gas_inlet',
                            {'nodes': ['gas_source', 'R0']}),
                           ('gas_outlet',
                            {'nodes': ['R' + str(nCSTRs), 'gas_sink']})
                         ])
    # Internal streams
    for nCSTR in range(nCSTRs - 1): # One less internal stream than the number of CSTRs
        source_name = 'R' + str(nCSTR)
        sink_name = 'R' + str(nCSTR+1)
        stream_name = source_name + '_to_' + sink_name
        stream[stream_name] = {'nodes': [source_name, sink_name]}
    return nodes, streams

def cstr_train_mass_flow_rates(network):
    """
    This function calculates the mass flow rates between each CSTR in the
    series.

    Parameters
    ----------
    network = a network object; this is passed automatically when the mass
        flow rates function is assigned to the network object

    Returns
    -------
    network = an updated network object
    """

    # Perturb the inlet parameters
    phase_g = getattr(params, 'phase_g', None)
    if phase_g is None:
        phase_g = 0
    op_space_params = getattr(params, 'op_space_params', None)
    if op_space_params is not None:
        network.perturb_stream_properties(op_space_params)

    # Perturb the inlet parameters
    if op_space_params is not None:
        network.perturb_stream_properties(op_space_params)

    # Get the mass flow rate of the inlet stream
    inlet_stream_name = 'gas_inlet'
    inlet_stream_idx = network.get_stream_idx(inlet_stream_name)
    mass_flow_gas_in = network.streams[inlet_stream_idx].mass_flow

    # Set the gas properties for the streams leaving each reactor in the series
    for nCSTR in range(nCSTRs):
        # Node name, ID number, and index
        cstr_name = 'R' + str(nCSTR)
        cstr_idx = network.get_node_idx(cstr_name)

        # Exit stream name, ID number, and index
        if nCSTR < nCSTRs:
            # General case, internal stream
            stream_name = 'R' + str(nCSTR) + '_to_R' + str(nCSTR+1)
        elif nCSTR == nCSTRs:
            # Special case, last stream to outlet
            stream_name = 'gas_outlet'
        stream_idx = network.get_stream_idx(stream_name)

        # Set the stream properties
        network.streams[stream_idx].T = network.nodes[cstr_idx].T
        network.streams[stream_idx].P = network.nodes[cstr_idx].P
        network.streams[stream_idx].mass_fracs = (
            network.nodes[cstr_idx].mass_fracs[phase_g])
        network.streams[stream_idx].find_rho()
        network.streams[stream_idx].mass_flow = mass_flow_gas_in
        network.streams[stream_idx].find_vol_flow()
