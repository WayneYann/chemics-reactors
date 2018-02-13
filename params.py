"""This module is for defining all of the parameter distributions and actual
parameters for the kinetics, mass flow rate functions, and anything else the
user wants to manipulate on-the-fly. The variables in this module are actually
set in the input module, and the variables here are just to define some of what
is possible. It is possible to create other variables here other than what is
initially included via setattr(...). Possible uses of this ability are defining
other variables that are needed by specific user-defined functions but that
should not otherwise be perturbed or altered."""

from collections import OrderedDict, namedtuple

# Define a named tuple to keep the uncertainty ranges for the parameters. The
# named tuple has the following fields:
#   'base': the baseline value
#   'a'   : the lower bound (for uniform/log-uniform distributions) or the mean
#           (for normal/log-normal distributions)
#   'b'   : the upper bound (for uniform/log-uniform distributions) or the
#           standard deviation (for normal/log-normal distributions)
#   'type': the distribution type, which is one of:
#               'u' -- uniform
#               'lu' -- log-uniform
#               'n' -- normal
#               'ln' -- log-normal
param_dist = namedtuple('param_dist', ['base', 'a', 'b', 'type'])

# Define the ordered dictionary with the distributions for each of the stream
# properties that is to be altered during a design of experiments run. Keys to
# the dict are space-delimited strings describing the stream properties. The
# first string is the stream name. The second string is the property name.
# Scalar properties only need these two strings. Species-based properties (e.g.,
# mw, mass_fracs, etc.) also need the species name as the third string. For mass
# fractions, the first mass fraction is always calculated from the others to
# ensure the mass fractions sum to 1.
op_space_params_dist = OrderedDict([])

# This dictionary holds the current values of the operating conditions being
# varied during a design of experiments run.
op_space_params = {}

# This is the combined parameter distribution dictionary. Keys should correspond
# to the names of the dictionaries with the *current* values of the
# corresponding parameters. This is the container for any parameters that should
# be accessible to the UQ/GSA module.
param_dists = OrderedDict([])
