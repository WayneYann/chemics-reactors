""" This is a file for specifying all of the network parameters as a series of
dicts and dicts-of-dicts, etc. These data structures are then used to initialize
the actual objects in the reactor network.  """

from copy import deepcopy
import numpy as np
from math import pi
import params
import reactor_models
import model_funcs

# Flag to use MPI or not
use_mpi = False

# Output filenames -- allowed keys are 'reactor', 'stream', and 'ty'
#   'reactor' has detailed information about the reactors
#   'stream' has detailed information about the streams
#   'ty' has raw t, _y_ data for judging convergence to steady state
output_filenames = {'Reactor': 'reactor_out.txt',
                    'Stream': 'stream_out.txt',
                    'ty': 'ty_out.txt'}

# Time clock settings
t_initial = 0.0 # Start time, s
t_final = 1000000.0 # Stop time, s
t_res = 1000.0 # Time resolution for output, s

# Solver settings
solver = 'vode'
solver_settings = {
                   'atol': 1e-8,
                   'rtol': 1e-10,
                   'lband': None,
                   'uband': None,
                   'method': 'bdf',
                   'with_jacobian': True,
                   'nsteps': 5000,
                   'first_step': 1e-8,
#                   'min_step': 1e-10,
#                   'max_step': 1,
                   'order': 5}

# CSTR train parameters
nCSTRs = 100 # Total number of CSTRs, adjust as needed
bed_diam = 0.025 # Bed diameter in m, adjust as needed
bed_length = 0.02 # Bed length in m, adjust as needed

# Create CSTR train object to help initialize the nodes and streams ordered
# dictionaries in the input module
nodes, streams = reactor_models.cstr_train(bed_diam, bed_length,
    num_CSTRs=nCSTRs, asPFR=True)

# Define or redefine the mass flow rates function in the model_funcs module
setattr(model_funcs, 'get_mass_flow_rates',
    reactor_models.cstr_train_mass_flow_rates)

# Summary of gas stream labels -- we only have gas streams, so this is simply
# the keys in the ordered dictionary
g_streams = [stream_name for stream_name in stream]

# Gas inlet and bed specifications
T_gi = 773.0            # inlet temperature, K
v_gi = 20.0 / 60.0 / 1000.0 # inlet flow rate, m^3/s at STP (from L/min)
P_gi = 101.325          # pressure after the distributor, kPa
mu_gi = 3.6400165e-5    # dynamic inlet viscosity, kg/m/s (for air at 773 K)
species_g = ['i', 'a', 'b', 'c', 'd'] # Adjust as needed, first is always inert!
MW_gi = np.array([1.0, 2.0, 3.0, 3.0, 2.0]) # molecular weights (g/mol)
mass_fracs_g = np.array([0.8, 0.1, 0.1, 0.0, 0.0]) # Mass fractions
eps_g = 0.46 # Close-packed void fraction in bed
phase_g = 0
v_gi = v_gi * (T_gi / 273.15) * (101.325 / P_gi) # m^3/s at T_gi, P_gi
stream_properties_g = {'T': T_gi, 'vol_flow': v_gi, 'P': P_gi, 'mw': MW_gi,
                       'species': species_g, 'mass_fracs': mass_fracs_g,
                       'mu': mu_gi, 'phase': phase_g}
reactor_properties_g = {'species': species_g, 'mass_fracs': mass_fracs_g,
                        'mu': mu_gi, 'mw': MW_gi, 'eps': eps_g}

# Gas stream initialization to ensure consistent phase numbers, etc. everywhere.
# The actual physical properties and compositions will be updated later (or can
# be manually specified).
for stream in g_streams:
    streams[stream]['properties'] = deepcopy(stream_properties_g)

# Bed solids specifications
species_s = ['i', 'c'] # first is inert, second is catalyst
d_p = [1e-5, 1e-5] # particle diameter, m
rho_s = [2000.0, 2000.0] # particle density, kg/m^3
MW_s = [1000.0, 1000.0] # Molecular weights, g/mol
eps_s = 1 - eps_g # Solids volume fraction from void fraction
phase_s = 1 # Index to solids phase
reactor_properties_s = {'species': species_s, 'mass_fracs': mass_fracs_s,
                        'mw': MW_s, 'dp': d_p, 'rho': rho_s, 'eps': eps_s}

# Add phase specifications for each reactor (initial conditions)
for nCSTR in range(nCSTRs):
    cstr_name = 'R' + str(nCSTR)
    nodes[cstr_name]['T'] = T_gi * 1
    nodes[cstr_name]['P'] = P_gi * 1
    nodes[cstr_name]['phases'] = {
            # Gas phase
            phase_g: deepcopy(reactor_properties_g)

            # Solids phase
            phase_s: deepcopy(reactor_properties_s)
            }
        # Energy balance parameters for each reactor (currently isothermal)
        nodes[cstr_name]['Q_balance_params'] = {'balance_type': 'isothermal'}
        }

# UQ/GSA settings
uq_gsa_settings = {
    #   None (no UQ/GSA)
    #   UQ (UQ only)
    #   VGSA1 (VGSA type 1, for first order and total effect indices only)
    #   VGSA2 (VGSA type 2, for first order, second order, and total effect indices)
    #   DOE (Design of Experiments, for a set of operating conditions all at the
    #   same set of kinetic parameters)
    'type': None,
#    'type': 'DOE',
#    'type': 'UQ',
#    'type': 'VGSA2',

    # Number of sample groups
    'replicates': 1,

    # Type of experimental design:
    #   full2: full two level factorial (pyDOE ff2n)
    #   frac: fractional factorial (pyDOE fracfact)
    #   pb: Plackett-Burman (pyDOE pbdesign)
    #   bb: Box-Behnken (pyDOE bbdesign)
    #   cc: central composite (pyDOE ccdesign)
    #   lh: Latin hypercube (pyDOE lhs)
    # Options are specified in a dictionary with the values required by the
    # corresponding pyDOE package. Most of them ('frac' excepted) only require
    # the number of desired conditions to run (automatically determined), and
    # defaults can be supplied for the remaining arguments. The 'frac' design
    # requires a requested alias structure as a string.
    'DOE_design': {
        'doe_args': {'type': 'cc',
                     'args': {'center': (0, 1), 'alpha': 'r', 'face': 'cci'}}
        },

    # What reactor/stream output do we want to track
    'output': {'Reactor': [],
               'Stream': []},
    'output_filenames': {'Reactor': 'reactor_uq_gsa_out',
                         'Stream': 'stream_uq_gsa_out'},

    # Random seeds for UQ/GSA from /dev/urandom
    'rand_seed': [],

    # Post-processing settings
    'ntraj': 1, # Number of independent trajectories to process
    'calculate_Sij': True,
    'UQ_percentiles': [5., 36., 50., 84., 95.]
    }

# Final check to make sure operating space parameters are not perturbed twice in
# a DOE run; otherwise very bad things could happen!
if uq_gsa_settings['type'] == 'DOE':
    # Remove the operating space parameter distributions from the general UQ/GSA
    # parameter distributions if present.
    params.param_dists.pop('op_space_params', None)
    # Put the parameter distribution into the DOE dictionary
    uq_gsa_settings['DOE_design']['doe_param_dict'] = 'op_space_params'
    uq_gsa_settings['DOE_design']['doe_param_dist'] = params.op_space_params_dist
else:
    # Reset the DOE design options to None if no DOE run is requested.
    uq_gsa_settings['DOE_design'] = None
