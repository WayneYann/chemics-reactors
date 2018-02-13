"""This module implements classes and functions related to carrying out
uncertainty quantification and global sensitivity analysis calculations."""

import random
import os
from math import log, exp
from copy import deepcopy
import numpy as np
import pyDOE
import sobol_seq

# TODO: Relax the assumption that all parameters are uncorrelated.
class uq_gsa_seq():
    """
    This object is used to generate points in parameter space for UQ/GSA
    calculations.
    """

    def __init__(self, param_dists, design_type=None, seed=None,
        doe_design=None):
        """
        This function initializes the object for generating the sequence of
        points used in UQ/GSA/DOE calculations.

        Parameters
        ----------
        param_dists = a dictionary of parameter distribution dictionaries; if
            this is a DOE, do *NOT* include the operating space dictionary or
            bad things will happen
        design_type = the kind of design sequence to follow; one of:
            'UQ' = basic uncertainty quantification (a series of independent
                vectors for each iteration)
            'VGSA1' = reduced-cost variance-based GSA (yields only first order
                and total sensitivity indices)
            'VGSA2' = full-cost variance-based GSA (yields first order, second
                order, and total sensitivity indices)
            'DOE' = a design of experiments run where certain operating
                conditions are systematically perturbed (only works for
                operating conditions, not other parameter types!)
        seed = the seed for the PRNG; if None, it is taken from os.urandom()
        doe_design = dictionary with the keys:
            'doe_args' = a dictionary with the arguments needed to construct the
                operating conditions sequence
            'doe_param_dict' = the name of the dictionary in the module that
                needs to be updated
            'doe_param_dist' = an ordered dictionary with the distributions for
                the operating conditions, used to generate an actual design
                matrix in non-coded units

        Returns
        -------
        uq_gsa_seq = an object for generating the sequences
        """

        # Basic data
        self.iteration = 0
        self.A_vector = None
        self.B_vector = None
        self.param_vector = None
        if design_type is None:
            print('No UQ/GSA design type specified. Defaulting to simple UQ.')
            self.design_type = 'UQ'
        elif design_type not in ['UQ', 'VGSA1', 'VGSA2', 'DOE']:
            raise ValueError('Invalid UQ/GSA type ' + design_type + ' selected')
        else:
            self.design_type = design_type

        # Parameter data
        self.param_dists = param_dists

        # Metadata about the parameter sets
        self.param_dists = param_dists
        param_list = [(pset, pval) for pset in param_dists
            for pval in param_dists[pset]]
        if self.design_type == 'DOE':
            pset = doe_design['doe_param_dict']
            param_list_doe = [(pset, pval) for pval in doe_design['doe_param_dist']]
        else:
            param_list_doe = []
        self.param_list = param_list
        self.param_list_doe = param_list_doe
        self.num_params = len(self.param_list) + len(self.param_list_doe)
        if design_type == 'UQ':
            self.max_iter = 1
        elif design_type == 'VGSA1':
            self.max_iter = self.num_params + 2
        elif design_type == 'VGSA2':
            self.max_iter = 2*self.num_params + 2
        elif design_type == 'DOE':
            if doe_design is None:
                raise ValueError('DOE requested with no design information supplied.')
            self.doe_design = doe_design
            self._generate_doe_param_list()
            self._generate_doe_design()
            self.max_iter = self.doe_design['doe_op_conds'].shape[0]

        # Random number generator
        self.random = random.Random()
        if seed is not None:
            self.random.seed(a=seed)

    def _generate_doe_param_list(self):
        """
        This function is responsible for generating a list of parameter names
        for DOE runs.

        Parameters
        ----------
        None

        Returns
        -------
        None, but sets the parameter list in the doe_design dictionary
        """

        param_list = [key for key in self.doe_design['doe_param_dist']]
        self.doe_design['doe_params'] = param_list

    def _generate_doe_design(self):
        """
        This function is responsible for generating the matrix of operating
        conditions used in the design of experiments. It supports all of the
        design types implemented by pyDOE with the exception of general full
        factorials (including mixed level factorials). Full factorials are only
        permitted to have two levels at the moment.

        Parameters
        ----------
        None

        Returns
        -------
        None, but updates the self object to have the needed auxiliary data
        """

        # Unpack the dictionary of DOE design parameters
        doe_type = self.doe_design['doe_args']['type']
        kwargs = self.doe_design['doe_args']['args']

        # Get the list of parameters to iterate over
        try:
            param_list = self.doe_design['doe_params']
        except:
            self._generate_doe_param_list()
            param_list = self.doe_design['doe_params']
        n = len(param_list)

        # Create the design matrix in coded units
        if doe_type == 'full2': # Two level general factorial
            coded_design_matrix = pyDOE.ff2n(n)
        elif doe_type == 'frac': # Fractional factorial
            gen = kwargs.pop('gen', None)
            if gen is None:
                raise ValueError('No generator sequence specified for a fractional factorial design.')
            coded_design_matrix = pyDOE.fracfact(gen)
        elif doe_type == 'pb': # Plackett-Burman
            coded_design_matrix = pyDOE.pbdesign(n)
        elif doe_type == 'bb': # Box-Behnken
            coded_design_matrix = pyDOE.bbdesign(n, **kwargs)
        elif doe_type == 'cc': # Central composite
            coded_design_matrix = pyDOE.ccdesign(n, **kwargs)
        elif doe_type == 'lh': # Latin hypercube
            coded_design_matrix = pyDOE.lhs(n, **kwargs)
        elif doe_type == 'sob': # Sobol' Lp-tau low discrepancy sequence
            samples = kwargs.pop('samples')
            coded_design_matrix = sobol_seq.i4_sobol_generate(n, samples)
        else:
            raise ValueError(
                'Unrecognized DOE design option ' + doe_type)

        # Convert the coded design matrix into an uncoded design matrix (i.e.,
        # in terms of the raw operating conditions). This takes the minimum and
        # maximum values from the distributions and places the points
        # accordingly.
        a_array = np.zeros(n) # Minimum
        b_array = np.zeros(n) # Maximum
        for i in range(n):
            pkey = param_list[i]
            a_array[i] = self.doe_design['doe_param_dist'][pkey].a
            b_array[i] = self.doe_design['doe_param_dist'][pkey].b
        r_array = b_array - a_array # Ranges for each dimension
        if doe_type in ['pb', 'bb', 'cc', 'frac', 'full2']:
            # These designs all have points clustered around a distinct center.
            # The coded matrix is in terms of unit perturbations from the
            # center, so we can just scale them by the ranges before adding the
            # center value. For these designs, we actually want to use half the
            # range so that the points that are supposed to be at the min/max
            # values end up there.
            c_array = (a_array + b_array)/2 # Center is average of min, max
            doe_op_conds = coded_design_matrix*r_array/2 + c_array 
        elif doe_type in ['lh', 'sob']:
            # The Latin hypercube and Sobol' sequences space points between a
            # range. This means the coded design matrix has all elements on
            # (0, 1). Since we don't have a center point, we place the operating
            # conditions with respect to the minimum values.
            doe_op_conds = coded_design_matrix*r_array + a_array 
        self.doe_design['doe_op_conds'] = doe_op_conds

    def get_doe_op_cond(self, dpt):
        """
        This function extracts the operating condition for the specified DOE
        design point.

        Parameters
        ----------
        dpt = an integer index specifying the design point to access

        Returns
        -------
        op_cond = a vector with the operating conditions for the design point
        """

        return self.doe_design['doe_op_conds'][dpt, :]

    def _random_number(self, a, b, t):
        """
        This function draws a random number from the appropriate distribution given
        the distribution parameters and type.

        Parameters
        ----------
        a = lower bound (uniform/log-uniform) or mean (normal/log-normal)
        b = upper bound (uniform/log-uniform) or standard deviation
            (normal/log-normal)
        t = string indicating distribution type as one of:
            'u' -- uniform
            'lu' -- log-uniform
            'n' -- normal
            'ln' -- log-normal

        Returns
        -------
        r = the random number
        """

        if t == 'u':
            r = self.random.uniform(a, b)
        elif t == 'lu':
            r = exp(self.random.uniform(log(a), log(b)))
        elif t == 'n':
            r = self.random.gauss(a, b)
        elif t == 'ln':
            r = self.random.lognormvariate(a, b)
        else:
            raise ValueError('Unknown probability distribution type ' + t)

        return r

    def generate_vectors(self, n):
        """
        This function generates the design vectors for carrying out the UQ/GSA
        calculations.

        Parameters
        ----------
        n = the current replicate in the sequence; only used if this is a DOE
            run to keep the other parameters at the base value if n = 0

        Returns
        -------
        self = an updated object with the selected design vectors
        """

        A_vector = {}
        B_vector = {}
        for pdist in self.param_dists:
            param_dist = self.param_dists[pdist]
            if self.design_type == 'DOE' and n == 0:
                # The A vector will just be the base parameter values
                A_vector[pdist] = {k: param_dist[k].base for k in param_dist}
            else:
                # Randomly sample the parameter values
                A_vector[pdist] = {k: self._random_number(param_dist[k].a,
                    param_dist[k].b, param_dist[k].type) for k in param_dist}
            if 'VGSA' in self.design_type:
                # Also need a B vector for VGSA calculations
                B_vector[pdist] = {k: self._random_number(param_dist[k].a,
                    param_dist[k].b, param_dist[k].type) for k in param_dist}
        self.A_vector = A_vector
        if 'VGSA' in self.design_type:
            self.B_vector = B_vector

    def perturb_params(self, module):
        """
        This function constructs the next design vector given the current
        iteration and sets it in the specified input module

        Parameters
        ----------
        module = a Python module with the parameters that need to be set

        Returns
        -------
        None, but sets the parameters in the module
        """

        # Generate the parameter vector. This puts the A and B vectors at the
        # end of the array to make post-processing easier. For UQ we only have a
        # series of A vectors.
        self.param_vector_doe = {}
        if self.design_type == 'UQ':
            self.param_vector = self.A_vector
        elif self.design_type == 'VGSA1':
            if self.iteration == self.num_params:
                self.param_vector = self.A_vector
                self.iteration += 1
            elif self.iteration == self.num_params + 1:
                self.param_vector = self.B_vector
                self.iteration = 0
            elif self.iteration < self.num_params:
                AB_vector = deepcopy(self.A_vector)
                pset, pval = self.param_list[self.iteration]
                AB_vector[pset][pval] = self.B_vector[pset][pval]
                self.param_vector = AB_vector
                self.iteration += 1
        elif self.design_type == 'VGSA2':
            if self.iteration == 2*self.num_params:
                self.param_vector = self.A_vector
                self.iteration += 1
            elif self.iteration == 2*self.num_params + 1:
                self.param_vector = self.B_vector
                self.iteration = 0
            elif self.iteration < self.num_params:
                AB_vector = deepcopy(self.A_vector)
                pset, pval = self.param_list[self.iteration]
                AB_vector[pset][pval] = self.B_vector[pset][pval]
                self.param_vector = AB_vector
                self.iteration += 1
            elif self.num_params-1 < self.iteration < 2*self.num_params:
                BA_vector = deepcopy(self.B_vector)
                pset, pval = self.param_list[self.iteration-self.num_params]
                BA_vector[pset][pval] = self.A_vector[pset][pval]
                self.param_vector = BA_vector
                self.iteration += 1
        elif self.design_type == 'DOE':
            # Always use the same parameter vector for every operating
            # condition
            self.param_vector = self.A_vector
            # Design of experiments, set the operating condition
            op_cond_dict = {}
            for op_cond_param in range(len(self.doe_design['doe_params'])):
                param_key = self.doe_design['doe_params'][op_cond_param]
                param_val = self.doe_design['doe_op_conds'][self.iteration, op_cond_param]
                op_cond_dict[param_key] = param_val
            self.param_vector_doe = op_cond_dict
            self.iteration += 1
            if self.iteration == self.max_iter:
                self.iteration = 0

        # Set the parameter values in the module
        for param_dict in self.param_vector:
            module_param_dict = getattr(module, param_dict, None)
            if module_param_dict is not None:
                for param in self.param_vector[param_dict]:
                    module_param_dict[param] = self.param_vector[param_dict][param]
                setattr(module, param_dict, module_param_dict)

        # Set the operating conditions if this is a DOE
        if self.design_type == 'DOE':
            param_dict = self.doe_design['doe_param_dict']
            module_param_dict = getattr(module, param_dict, None)
            if module_param_dict is not None:
                for param_key in op_cond_dict:
                    module_param_dict[param_key] = op_cond_dict[param_key]
                setattr(module, param_dict, module_param_dict)

    def calculate_UQ_metrics(self, data, pctile=None):
        """
        This function is responsible for actually calculating the UQ metrics for
        a given dataset. It assumes that the dataset is compatible with the
        UQ/GSA trajectory specified when the object was created.

        Parameters
        ----------
        data = an array with the data to process
        pctile = a list of percentiles to calculate; if None, default values of
            [5, 16, 50, 84, 95] are used, which correspond to the mean and +/-1
            and +/-2 s.d. for a normal distribution

        Returns
        -------
        pctile_out = an array of the values at the requested percentiles
        """

        # Reshape the array for better performance
        data_new = np.reshape(data, (-1, data.shape[-1]))

        # Set the percentiles
        if pctile is None:
            pct = [5., 16., 50., 84., 95.]
        else:
            pct = pctile

        # Calculate the actual results
        pctile_tmp = np.nanpercentile(data_new, pct, axis=0)
        pct = np.expand_dims(np.asarray(pct), axis=1)
        pctile_out = np.concatenate((pct, pctile_tmp), axis=1)
        return pctile_out

    def calculate_GSA_metrics(self, data, Sij=False):
        """
        This function is responsible for actually calculating the GSA metrics
        for a given dataset. It assumes that the dataset is compatible with the
        UQ/GSA trajectory specified when the object was created.

        Parameters
        ----------
        data = an array with the data to process
        Sij = flag for whether to calculate second order SI; defaults to False
            and only possible if a 'VGSA2' trajectory was selected

        Returns
        -------
        Si_out = a 2-d array of first order SI
        STi_out = a 2-d array of total order SI
        Sij_out = a 3-d array of second order SI
        """

        # Bail out if the dataset is for a UQ calculation
        if self.design_type == 'UQ':
            raise UserWarning('GSA calculation not possible for a UQ dataset')
            return None

        # Get the shape of the data array
        nproc, nrep, npert, nelem = data.shape

        # Accumulation arrays for partial variances for Si, STi (Vi, VTi)
        if self.design_type == 'VGSA1':
            Vi = np.zeros((self.num_params, nelem, 1))
            VTi = np.zeros((self.num_params, nelem, 1))
        elif self.design_type == 'VGSA2':
            Vi = np.zeros((self.num_params, nelem, 2))
            VTi = np.zeros((self.num_params, nelem, 2))
            if Sij:
                Vij = np.zeros((self.num_params, self.num_params, nelem, 2))

        # Carry out the partial sums for the current set of data
        data_new = np.reshape(data, (-1, nelem))
        idx = 0
        nrep_total = 0
        for rep_idx in range(nrep*nproc):
            if self.design_type == 'VGSA1':
                # Extract the data for the current replicate block
                f_A = data_new[self.num_params+idx, :]
                f_B = data_new[self.num_params+idx+1, :]
                f_AB = data_new[idx:self.num_params+idx, :]

                if not np.any(np.isnan(f_A)):
                    # Calculate the contributions to the partial variance
                    Vi_term = f_B * (f_AB - f_A)
                    Vi[:, :, 0] += Vi_term
                    VTi_term = (f_A - f_AB)**2
                    VTi[:, :, 0] += VTi_term
                    nrep_total += 1

                # Update the block index
                idx += (self.num_params+2)
            elif self.design_type == 'VGSA2':
                # Extract the data for the current replicate block
                f_A = data_new[2*self.num_params+idx, :]
                f_B = data_new[2*self.num_params+idx+1, :]
                f_AB = data_new[idx:self.num_params+idx, :]
                f_BA = data_new[self.num_params+idx:2*self.num_params+idx, :]

                if not np.any(np.isnan(f_A)):
                    # Calculate the contributions to the partial variance
                    Vi_term = f_B * (f_AB - f_A)
                    Vi[:, :, 0] += Vi_term
                    Vi_term = f_A * (f_BA - f_B)
                    Vi[:, :, 1] += Vi_term
                    VTi_term = (f_A - f_AB)**2
                    VTi[:, :, 0] += VTi_term
                    VTi_term = (f_B - f_BA)**2
                    VTi[:, :, 1] += VTi_term
                    nrep_total += 1

                    # Partial variances for Sij (Vij) -- NOTE: this is actually
                    # the matrix of 'closed' second order effects and is not yet
                    # the matrix of second order partial variances. To convert
                    # the closed second order partial variances to real second
                    # order partial variances, we need to subtract the first
                    # order partial variances. We will do this later.
                    if Sij:
                        # This calculation only needs to consider the elements
                        # in the upper triangular portion of the Vij matrix as
                        # the matrix is otherwise symmetric (Sij = Sji) and
                        # second order coefficients make no sense for the
                        # diagonal elements.
                        for i in range(self.num_params):
                            for j in range(i+1, self.num_params):
                                f_ABi = data_new[i+idx, :]
                                f_ABj = data_new[j+idx, :]
                                f_BAi = data_new[self.num_params+i+idx, :]
                                f_BAj = data_new[self.num_params+j+idx, :]
                                Vij_term = f_ABj*(f_BAi - f_BAj)
                                Vij[i, j, :, 0] += Vij_term
                                Vij_term = f_BAi*(f_ABj - f_ABi)
                                Vij[i, j, :, 1] += Vij_term

                # Update the block index
                idx += (2*self.num_params+2)

        # Normalize estimators to finish calculation of partial variances
        Vi /= nrep_total
        VTi /= (2*nrep_total)
        if Sij and self.design_type == 'VGSA2':
            # Normalize estimator
            Vij /= nrep_total

            # Remove first order partial variances to get true second order
            # partial variances, rather than the closed effect second order
            # partial variances.
            for i in range(self.num_params):
                for j in range(i+1, self.num_params):
                    Vij[i, j, :, :] -= (Vi[i, :, :] + Vi[j, :, :])

        # Total variances
        if self.design_type == 'VGSA1':
            # A data
            A_data = np.reshape(
                data[:, :, self.num_params::self.num_params+2, :], (-1, nelem))

            # B data
            B_data = np.reshape(
                data[:, :, self.num_params+1::self.num_params+2, :], (-1, nelem))

            # Variance for Si -- estimate 1 based on the B vector
            V1 = np.expand_dims(np.nanvar(B_data, axis=0), 1)

            # Variance for STi -- estimate 1 based on the A vector
            VT = np.expand_dims(np.nanvar(A_data, axis=0), 1)

        elif self.design_type == 'VGSA2':
            # A data
            A_data = np.reshape(
                data[:, :, 2*self.num_params::2*self.num_params+2, :], (-1, nelem))

            # B data
            B_data = np.reshape(
                data[:, :, 2*self.num_params+1::2*self.num_params+2, :], (-1, nelem))

            # Variance for Si -- estimate 1/2 based on the B/A vector
            V1 = np.zeros((nelem, 2))
            V1[:, 0] = np.nanvar(B_data, axis=0)
            V1[:, 1] = np.nanvar(A_data, axis=0)

            # Variance for STi -- estimate 1/2 based on the A/B vector
            VT = np.zeros((nelem, 2))
            VT[:, 0] = np.nanvar(A_data, axis=0)
            VT[:, 1] = np.nanvar(B_data, axis=0)

            # Variance for Sij
            if Sij:
                V2 = np.zeros((self.num_params, nelem, 2))
                for i in range(self.num_params):
                    AB_data = np.reshape(
                        data[:, :, i::2*self.num_params+2, :], (-1, nelem))
                    BA_data = np.reshape(
                        data[:, :, self.num_params+i::2*self.num_params+2, :], (-1, nelem))
                    V2[i, :, 0] = np.nanvar(AB_data, axis=0)
                    V2[i, :, 1] = np.nanvar(BA_data, axis=0)

        # Sensitivity indices from the ratios of the partial and total variances
        Si_out = Vi / V1
        STi_out = VTi / VT
        if Sij and self.design_type == 'VGSA2':
            Sij_out = Vij / V2

        # If needed, average the Si, STi, and Sij values for the VGSA2 method
        if self.design_type == 'VGSA1':
            Si_out = Si_out[:, :, 0]
            STi_out = STi_out[:, :, 0]
        elif self.design_type == 'VGSA2':
            Si_out = np.nanmean(Si_out, axis=2)
            STi_out = np.nanmean(STi_out, axis=2)
            if Sij:
                Sij_out = np.nanmean(Sij_out, axis=3)

        # Return SI data
        if not Sij:
            Sij_out = None
        return (Si_out, STi_out, Sij_out)
