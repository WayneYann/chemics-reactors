"""This module handles data output to disk."""
import reactor_network as rn
import numpy as np

class io_handler():
    """
    This class handles opening the data files and writing the output.
    """

    def __init__(self, base_files=None, uq_gsa_files=None, uq_gsa_output=None,
        purge=None, rank=None):
        """
        Initializes the data structures to store the data and the associated
        disk files.

        Parameters
        ----------
        base_files = dictionary of data types and associated filenames for output at
            the base case
        uq_gsa_files = dictionary of data types and associated filenames for output at
            related to UQ/GSA calculations
        purge = flag to determine whether to create a new file/delete existing
            contents on initialization

        Returns
        -------
        self = an I/O object
        """

        # Extension in the case of multiple MPI ranks
        if rank is None:
            ext = '.txt'
        else:
            ext = '_' + str(rank) + '.txt'
        self.filename_ext = ext

        # Base case results
        self.base_filenames = base_files
        if base_files is None:
            print('WARNING: No base case output will be saved')
        self.data = {'Reactor': None, 'Stream': None}
        self.names = {'Reactor': [], 'Stream': []}

        if base_files is not None:
            # Create empty files for the results
            if purge is None or purge:
                for file_type in base_files:
                    with open(base_files[file_type], 'wt') as f:
                        pass

        # UQ/GSA results
        self.uq_gsa_data = {'Reactor': None, 'Stream': None}
        self.uq_gsa_params = {'Reactor': None, 'Stream': None}
        self.uq_gsa_names = {'Reactor': [], 'Stream': []}
        self.num_uq_gsa_responses = {'Reactor': 0, 'Stream': 0}
        self.uq_gsa_filenames = uq_gsa_files
        self.uq_gsa_output = uq_gsa_output
        self.uq_gsa_header_written = False
        if uq_gsa_files is not None:
            # Create empty files for the results
            if purge is None or purge:
                for file_type in uq_gsa_files:
                    if self.uq_gsa_output[file_type]:
                        with open(uq_gsa_files[file_type] + ext, 'wt') as f:
                            pass

    def write_y(self, t, network, y_func):
        """
        This is a function to write the t, _y_ data to disk to aid in judging
        convergence to steady state. Don't try to use this data for anything
        else, though, as it is not nicely formatted!

        Parameters
        ----------
        t = the current time
        network = a network object from which to extract the solution vector
        y_func = a function used to extract the _y_ vector from the network
            object

        Returns
        -------
        Nothing, but writes t, _y_ to disk
        """

        y = y_func(t, network)

        fmt = '{:13.6E} '*(len(y)+1) + '\n'
        with open(self.base_filenames['ty'], 'at') as ty:
            ty.write(fmt.format(t, *y))

    def get_data(self, network):
        """
        This function extracts the data to be written to disk from the network
        object.

        Parameters
        ----------
        network = the network object with the data to be extracted

        Returns
        -------
        none, but sets the data fields in the IO object in preparation for
            writing it to disk
        """

        # Get the number of reactors, the total number of species across all
        # phases, and the number of streams
        num_reactors = network.num_reactors
        num_phases = max(network.num_phases)
        num_streams = network.num_streams

        # Dictionaries for output data
        rxtr_data = {i: {} for i in range(num_phases)}
        stream_data = {i: {} for i in range(num_phases)}

        # Create a list of the reactor and stream indices
        reactor_idxs = [node_idx for node_idx in range(len(network.nodes)) if
            isinstance(network.nodes[node_idx], rn.Reactor)]
        stream_idxs = list(range(len(network.streams)))

        for phase_idx in range(num_phases):

            # Create a list of all the reactor nodes in the network that have
            # this phase
            num_species = np.amax(network.num_species_per_phase[phase_idx, :])

            # Only extract data if the phase is actually present in the reactors
            if num_species > 0:

                # Initialize arrays for the data for this phase in all reactors
                T = np.zeros(num_reactors) * np.nan # Temperature
                P = np.zeros(num_reactors) * np.nan # Pressure
                mu = np.zeros(num_reactors) * np.nan # Viscosity
                rho = np.zeros(num_reactors) * np.nan # Density
                eps = np.zeros(num_reactors) * np.nan # Volume fraction
                mass = np.zeros(num_reactors) * np.nan # Total mass
                mass_fracs = np.zeros((num_species, num_reactors)) * np.nan # Mass fractions
                dp = np.zeros((num_species, num_reactors)) * np.nan # Particle diameters

                # Extract the data for the reactors
                species = None
                phase_type = None
                for ridx in range(len(reactor_idxs)):
                    num_species_ridx = network.num_species_per_phase[phase_idx, ridx]
                    if num_species_ridx > 0:
                        node = network.nodes[reactor_idxs[ridx]]
                        if species is None:
                            species = node.species[phase_idx] # Species names
                        if phase_type is None:
                            phase_type = node.phase_type[phase_idx]
                        ns = len(node.mass_fracs[phase_idx])
                        T[ridx] = node.T
                        P[ridx] = node.P
                        if node.phase_type[phase_idx] != 's':
                            # Only set viscosity for gases/liquids
                            mu[ridx] = node.mu[phase_idx]
                        rho[ridx] = node.rho_avg[phase_idx]
                        eps[ridx] = node.eps[phase_idx]
                        mass[ridx] = node.masses[phase_idx]
                        mass_fracs[:ns, ridx] = node.mass_fracs[phase_idx]
                        if node.phase_type[phase_idx] == 's':
                            # Only extract particle diameters for solid phases
                            dp[:ns, ridx] = node.dp[phase_idx]
                    else:
                        # No phase of this type in this reactor
                        pass

            else:

                # Everything is empty
                T = None
                P = None
                mu = None
                rho = None
                eps = None
                mass = None
                mass_fracs = None
                dp = None
                species = None
                phase_type = None

            # Assign the reactor data to the dictionary
            rxtr_data[phase_idx] = {'T': T,
                                    'P': P,
                                    'phase_type': phase_type,
                                    'species': species,
                                    'mu': mu,
                                    'rho': rho,
                                    'eps': eps,
                                    'mass': mass,
                                    'mass_fracs': mass_fracs,
                                    'dp': dp}

            # Only create arrays if this phase is present in the streams
            if (any((network.streams[stream_idx].phase == phase_idx)
                for stream_idx in stream_idxs)):

                # Initialize arrays for data for this phase and all streams -- don't
                # need volume fraction since all streams are assumed to be of a
                # single phase
                T = np.zeros(num_streams) * np.nan # Temperature
                P = np.zeros(num_streams) * np.nan # Pressure
                mu = np.zeros(num_streams) * np.nan # Viscosity
                rho = np.zeros(num_streams) * np.nan # Density
                mass = np.zeros(num_streams) * np.nan # Total mass flow
                mass_fracs = np.zeros((num_species, num_streams)) * np.nan # Mass fractions
                dp = np.zeros((num_species, num_streams)) * np.nan # Particle diameters

                # Extract the data for the streams that have this phase
                species = None
                phase_type = None
                for sidx in stream_idxs:
                    if network.streams[sidx].phase == phase_idx:
                        stream = network.streams[sidx]
                        if species is None:
                            species = stream.species # Species names
                        if phase_type is None:
                            phase_type = stream.phase_type
                        ns = len(stream.mass_fracs)
                        T[sidx] = stream.T
                        P[sidx] = stream.P
                        # Only set viscosity for gases/liquids
                        if phase_type != 's':
                            mu[sidx] = stream.mu
                        rho[sidx] = stream.rho_avg
                        mass[sidx] = stream.mass_flow
                        mass_fracs[:ns, sidx] = stream.mass_fracs
                        # Only extract particle diameters for solid phases
                        if phase_type == 's':
                            dp[:ns, sidx] = stream.dp
                    else:
                        # This stream does not have this phase
                        pass

            else:

                # Everything is empty
                T = None
                P = None
                mu = None
                rho = None
                mass = None
                mass_fracs = None
                dp = None
                species = None
                phase_type = None

            # Assign the stream data to the dictionary
            stream_data[phase_idx] = {'T': T,
                                      'P': P,
                                      'phase_type': phase_type,
                                      'species': species,
                                      'mu': mu,
                                      'rho': rho,
                                      'mass_flow': mass,
                                      'mass_fracs': mass_fracs,
                                      'dp': dp}

        # Assign the collected data to the return object
        self.data = {'Reactor': rxtr_data, 'Stream': stream_data}

        # Assign the names of the reactors and streams to the names dictionary
        self.names['Reactor'] = [network.nodes[ridx].name for ridx in
            range(len(network.nodes)) if isinstance(network.nodes[ridx],
            rn.Reactor)]
        self.names['Stream'] = [network.streams[sidx].name for sidx in
            range(len(network.streams))]

    def write_output(self, t):
        """
        This function will write output from the network at the current time to
        the specified files.

        Parameters
        ----------
        t = the current simulation time (s)

        Returns
        -------
        None
        """

        for file_type in ['Reactor', 'Stream']:
            filename = self.base_filenames[file_type]
            with open(filename, 'at') as f:
                f.write('{:s} results at t = {:13.6E} s\n'.format(file_type, t))
                if file_type == 'Reactor':
                    f.write('\nReactor indices and names\n')
                    for i in range(len(self.names['Reactor'])):
                        f.write(' {:d} {:s}\n'.format(i, self.names['Reactor'][i]))
                    for phase in self.data[file_type]:
                        if self.data[file_type][phase]['phase_type'] == 'g':
                            phase_type = 'gas'
                        elif self.data[file_type][phase]['phase_type'] == 's':
                            phase_type = 'solids'
                        elif self.data[file_type][phase]['phase_type'] == 'l':
                            phase_type = 'liquid'
                        f.write('\nScalar quantities for ' + phase_type +
                            ' phase ' + str(phase) + '\n')
                        num_reactors = self.data[file_type][phase]['T'].size
                        f.write(' '*30 + ' Reactor ID\n')
                        fmt = '{:30s} ' + ' {:13d}'*num_reactors + '\n'
                        f.write(fmt.format('Quantity', *range(num_reactors)))
                        fmt = '{:30s} ' + ' {:13.6E}'*num_reactors + '\n'
                        f.write(fmt.format(
                            'Temperature [K]', *self.data[file_type][phase]['T']))
                        f.write(fmt.format(
                            'Pressure [kPa]', *self.data[file_type][phase]['P']))
                        f.write(fmt.format(
                            'Vol. Fraction []', *self.data[file_type][phase]['eps']))
                        f.write(fmt.format(
                            'Density [kg/m^3]', *self.data[file_type][phase]['rho']))
                        f.write(fmt.format(
                            'Viscosity [kg/m/s]', *self.data[file_type][phase]['mu']))
                        f.write(fmt.format(
                            'Total mass [kg]', *self.data[file_type][phase]['mass']))
                        num_species = len(self.data[file_type][phase]['species'])
                        f.write('\nMass fractions for ' + phase_type +
                            ' phase ' + str(phase) + '\n')
                        f.write(' '*30 + ' Reactor ID\n')
                        fmt = '{:30s} ' + ' {:13d}'*num_reactors + '\n'
                        f.write(fmt.format('Species', *range(num_reactors)))
                        fmt = '{:30s} ' + ' {:13.6E}'*num_reactors + '\n'
                        for s_idx in range(num_species):
                            species_name = self.data[file_type][phase]['species'][s_idx]
                            f.write(fmt.format(species_name,
                            *self.data[file_type][phase]['mass_fracs'][s_idx, :]))
                        if phase_type == 'solids':
                            f.write('\nParticle diameters for ' + phase_type +
                                ' phase ' + str(phase) + '\n')
                            f.write(' '*30 + ' Reactor ID\n')
                            fmt = '{:30s} ' + ' {:13d}'*num_reactors + '\n'
                            f.write(fmt.format('Species', *range(num_reactors)))
                            fmt = '{:30s} ' + ' {:13.6E}'*num_reactors + '\n'
                            for s_idx in range(num_species):
                                species_name = self.data[file_type][phase]['species'][s_idx]
                                f.write(fmt.format(species_name,
                                *self.data[file_type][phase]['dp'][s_idx, :]))

                elif file_type == 'Stream':
                    f.write('\nStream indices and names\n')
                    for i in range(len(self.names['Stream'])):
                        f.write(' {:d} {:s}\n'.format(i, self.names['Stream'][i]))
                    for phase in self.data[file_type]:
                        if self.data[file_type][phase]['phase_type'] == 'g':
                            phase_type = 'gas'
                        elif self.data[file_type][phase]['phase_type'] == 's':
                            phase_type = 'solids'
                        elif self.data[file_type][phase]['phase_type'] == 'l':
                            phase_type = 'liquid'
                        f.write('\nScalar quantities for ' + phase_type +
                            ' phase ' + str(phase) + '\n')
                        num_streams = self.data[file_type][phase]['T'].size
                        f.write(' '*30 + ' Stream ID\n')
                        fmt = '{:30s} ' + ' {:13d}'*num_streams + '\n'
                        f.write(fmt.format('Quantity', *range(num_streams)))
                        fmt = '{:30s} ' + ' {:13.6E}'*num_streams + '\n'
                        f.write(fmt.format(
                            'Temperature [K]', *self.data[file_type][phase]['T']))
                        f.write(fmt.format(
                            'Pressure [kPa]', *self.data[file_type][phase]['P']))
                        f.write(fmt.format(
                            'Density [kg/m^3]', *self.data[file_type][phase]['rho']))
                        f.write(fmt.format(
                            'Viscosity [kg/m/s]', *self.data[file_type][phase]['mu']))
                        f.write(fmt.format(
                            'Total mass flow [kg/s]', *self.data[file_type][phase]['mass_flow']))
                        num_species = len(self.data[file_type][phase]['species'])
                        f.write('\nMass fractions for ' + phase_type +
                            ' phase ' + str(phase) + '\n')
                        f.write(' '*30 + ' Stream ID\n')
                        fmt = '{:30s} ' + ' {:13d}'*num_streams + '\n'
                        f.write(fmt.format('Species', *range(num_streams)))
                        fmt = '{:30s} ' + ' {:13.6E}'*num_streams + '\n'
                        for s_idx in range(num_species):
                            species_name = self.data[file_type][phase]['species'][s_idx]
                            f.write(fmt.format(species_name,
                                *self.data[file_type][phase]['mass_fracs'][s_idx, :]))
                        if phase_type == 'solids':
                            f.write('\nParticle diameters for ' + phase_type +
                                ' phase ' + str(phase) + '\n')
                            f.write(' '*30 + ' Stream ID\n')
                            fmt = '{:30s} ' + ' {:13d}'*num_streams + '\n'
                            f.write(fmt.format('Species', *range(num_streams)))
                            fmt = '{:30s} ' + ' {:13.6E}'*num_streams + '\n'
                            for s_idx in range(num_species):
                                species_name = self.data[file_type][phase]['species'][s_idx]
                                f.write(fmt.format(species_name,
                                    *self.data[file_type][phase]['dp'][s_idx, :]))

    def init_uq_gsa_data(self, network, uq_gsa_seq, nrep, nvec):
        """
        This function initializes the data arrays for the UQ/GSA data at each
        phase point.

        Parameters
        ----------
        network = the network object with the data to be extracted
        uq_gsa_seq = a UQ/GSA sequence object for getting the parameter names
        nrep = the number of replicates in the trajectory
        nvec = the number of vectors in this local block

        Returns
        -------
        none, but initializes the data fields
        """

        # Loop over each data type and initialize the vectors and header labels
        for file_type in ['Reactor', 'Stream']:
            uq_names = self.uq_gsa_output[file_type]
            if not uq_names:
                # There is no output requested for this type of output; move to
                # the next type
                continue
            param_list = uq_gsa_seq.param_list_doe + uq_gsa_seq.param_list
            self.param_names = [param[1] for param in param_list]
            if file_type == 'Reactor':
                reactor_names = []
                for reactor_name in uq_names:
                    reactor_id = network.node_name_map[reactor_name]
                    reactor_idx = network.node_map[reactor_id]
                    reactor_names_tmp = []
                    reactor_names_tmp.append('Temperature [K]')
                    reactor_names_tmp.append('Pressure [kPa]')
                    num_phases = network.nodes[reactor_idx].num_phases
                    for i in range(num_phases):
                        for species in network.nodes[reactor_idx].species[i]:
                            reactor_names_tmp.append(species + '_' + str(i) + ' [kg]')
                    reactor_names.append(reactor_names_tmp)
                num_reactor_elements = sum(len(reactor_names[i]) for i in
                    range(len(reactor_names)))
                self.uq_gsa_data['Reactor'] = np.zeros((nrep, nvec, num_reactor_elements))
                self.uq_gsa_params['Reactor'] = np.zeros((nrep, nvec, len(param_list)))
                self.num_uq_gsa_responses['Reactor'] = num_reactor_elements
                self.uq_gsa_names['Reactor'] = reactor_names
            elif file_type == 'Stream':
                # Initialize the name lists and data vector
                stream_names = []
                for stream_name in uq_names:
                    stream_id = network.stream_name_map[stream_name]
                    stream_idx = network.stream_map[stream_id]
                    stream_names_tmp = []
                    stream_names_tmp.append('Temperature [K]')
                    stream_names_tmp.append('Pressure [kPa]')
                    for species in network.streams[stream_idx].species:
                        stream_names_tmp.append(species + ' [kg/s]')
                    stream_names.append(stream_names_tmp)
                num_stream_elements = sum(len(stream_names[i]) for i in
                    range(len(stream_names)))
                self.uq_gsa_data['Stream'] = np.zeros((nrep, nvec, num_stream_elements))
                self.uq_gsa_params['Stream'] = np.zeros((nrep, nvec, len(param_list)))
                self.num_uq_gsa_responses['Stream'] = num_stream_elements
                self.uq_gsa_names['Stream'] = stream_names

    def set_uq_gsa_data(self, network, uq_gsa_seq, ridx, vidx, failure=False):
        """
        This function extracts the UQ/GSA data to be written to disk from the
        network object and sets it in the IO object.

        Parameters
        ----------
        network = the network object with the data to be extracted
        uq_gsa_seq = a UQ/GSA sequence object for getting the parameter values
        ridx = the index denoting which replicate the vector is for
        vidx = the index denoting which vector in the sequence needs to be set
        failure = a boolean to flag whether to set the resulting data as
            nonsense

        Returns
        -------
        none, but sets the data fields in the IO object in preparation for
            writing it to disk
        """

        # Loop over each data type and get the data
        for file_type in ['Reactor', 'Stream']:
            uq_names = self.uq_gsa_output[file_type]
            if not uq_names:
                # There is no output requested for this type of output; move to
                # the next type
                continue

            # Put the perturbed parameters into the parameter array
            param_vector = [uq_gsa_seq.param_vector_doe[pval[1]]
                for pval in uq_gsa_seq.param_list_doe]
            param_vector += [uq_gsa_seq.param_vector[pset][pval]
                for pset, pval in uq_gsa_seq.param_list]
            param_vector = np.asarray(param_vector)

            if file_type == 'Reactor':
                # Put the data into the data array
                idx = 0
                if not failure:
                    for reactor_name in uq_names:
                        reactor_id = network.node_name_map[reactor_name]
                        reactor_idx = network.node_map[reactor_id]
                        T = network.nodes[reactor_idx].T
                        P = network.nodes[reactor_idx].P
                        self.uq_gsa_data['Reactor'][ridx, vidx, idx] = T
                        self.uq_gsa_data['Reactor'][ridx, vidx, idx+1] = P
                        num_phases = network.nodes[reactor_idx].num_phases
                        idx += 2
                        for j in range(num_phases):
                            mass = network.nodes[reactor_idx].masses[j]
                            mass_fracs = network.nodes[reactor_idx].mass_fracs[j]
                            nspec = len(mass_fracs)
                            self.uq_gsa_data['Reactor'][ridx, vidx, idx:idx+nspec] = mass*mass_fracs
                            idx += nspec
                else:
                    self.uq_gsa_data['Reactor'][ridx, vidx, :] = np.nan

                self.uq_gsa_params['Reactor'][ridx, vidx, :] = param_vector

            elif file_type == 'Stream':
                # Put the data into the data array
                idx = 0
                if not failure:
                    for stream_name in uq_names:
                        stream_id = network.stream_name_map[stream_name]
                        stream_idx = network.stream_map[stream_id]
                        T = network.streams[stream_idx].T
                        P = network.streams[stream_idx].P
                        mass_flow = network.streams[stream_idx].mass_flow
                        mass_fracs = network.streams[stream_idx].mass_fracs
                        nspec = len(mass_fracs)
                        self.uq_gsa_data['Stream'][ridx, vidx, idx] = T
                        self.uq_gsa_data['Stream'][ridx, vidx, idx+1] = P
                        self.uq_gsa_data['Stream'][ridx, vidx, idx+2:idx+nspec+2] = mass_flow*mass_fracs
                        idx += (nspec + 2)
                else:
                    self.uq_gsa_data['Stream'][ridx, vidx, :] = np.nan

                self.uq_gsa_params['Stream'][ridx, vidx, :] = param_vector

    def save_uq_gsa_data(self, rank=None):
        """
        This function will save the UQ/GSA data to a NumPy npy file for later
        retrieval.

        Parameters
        ----------
        rank = the rank of the process saving the data; if no rank is specified,
            a serial trajectory is assumed

        Returns
        -------
        none, but saves the array(s) to disk
        """

        if rank is None:
            fext = '.npy'
        else:
            fext = '_' + str(rank) + '.npy'
        for file_type in ['Reactor', 'Stream']:
            if file_type == 'Reactor':
                fname = 'uq_gsa_reactor_data' + fext
                np.save(fname, self.uq_gsa_data['Reactor'])
                fname = 'uq_gsa_reactor_params' + fext
                np.save(fname, self.uq_gsa_params['Reactor'])
            elif file_type == 'Stream':
                fname = 'uq_gsa_stream_data' + fext
                np.save(fname, self.uq_gsa_data['Stream'])
                fname = 'uq_gsa_stream_params' + fext
                np.save(fname, self.uq_gsa_params['Stream'])

    def write_uq_gsa_output(self, t, rep_no, npert):
        """
        This function will write the output for UQ/GSA calculations.

        Parameters
        ----------
        t = the current simulation time (s)
        rep_no = the number of the current replicate
        pert_no = the number of the current perturbation vector within the
            replicate block

        Returns
        -------
        None, but writes the data to disk
        """

        # Write data, including headers if needed
        for file_type in ['Reactor', 'Stream']:
            filename = self.uq_gsa_filenames[file_type] + self.filename_ext
            uq_names = self.uq_gsa_output[file_type]
            if not uq_names:
                # There is no output requested for this type of output; move to
                # the next type
                continue
            with open(filename, 'at') as f:
                if file_type == 'Reactor':
                    # Create and write header
                    if not self.uq_gsa_header_written:
                        f.write('UQ/GSA results for reactor data\n\n')
                        f.write('Perturbed parameter values (see input module for units and values)\n')
                        f.write('Temperatures [K], pressures [kPa], and species masses [kg]\n\n')
                        reactor_header1 = '{:8s} {:8s}'.format(' ', ' ')
                        reactor_header2 = '{:8s} {:8s}'.format('Rep.', 'Pert.')
                        names = self.uq_gsa_names[file_type]
                        num_elements = [len(name_group) for name_group in names]
                        names_tmp = self.param_names
                        strw = [max(16, len(s)) for s in names_tmp]
                        nvals = len(names_tmp)
                        for i in range(nvals):
                            w = strw[i]
                            name_tmp = names_tmp[i]
                            reactor_header2 += (' {:' + str(w) +'s}').format(name_tmp)
                        reactor_header1 += (
                            (' {:' + str(len(reactor_header2)-18) + 's}').format(
                            'Parameter values'))
                        names_tmp = [name for name_group in names for name in name_group]
                        nvals = len(names_tmp)
                        reactor_header2 += (' {:16s}'*nvals).format(*names_tmp)
                        for name_idx in range(len(num_elements)):
                            reactor_header1 += (' {:' +
                                str(num_elements[name_idx]*17-1) + 's}').format(
                                uq_names[name_idx])
                        f.write(reactor_header1 + '\n')
                        f.write(reactor_header2 + '\n')
                    for pert_no in range(npert):
                        uq_gsa_params = self.uq_gsa_params[file_type][rep_no, pert_no, :]
                        uq_gsa_data = self.uq_gsa_data[file_type][rep_no, pert_no, :]
                        data_str_fmt = '{:8d} {:8d}'
                        nvals = len(uq_gsa_params)
                        strw = [max(16, len(s)) for s in self.param_names]
                        for i in range(nvals):
                            data_str_fmt += (' {:' + str(strw[i]) +'.6E}')
                        nvals = len(uq_gsa_data)
                        data_str_fmt += (' {:16.6E}')*nvals
                        data_str = data_str_fmt.format(rep_no, pert_no,
                            *uq_gsa_params, *uq_gsa_data)
                        f.write(data_str + '\n')
                elif file_type == 'Stream':
                    # Create and write header
                    if not self.uq_gsa_header_written:
                        f.write('UQ/GSA results for stream data\n\n')
                        f.write('Perturbed parameter values (see input module for units and values)\n')
                        f.write('Temperatures [K], pressures [kPa], and species mass flow rates [kg/s]\n\n')
                        stream_header1 = '{:8s} {:8s}'.format(' ', ' ')
                        stream_header2 = '{:8s} {:8s}'.format('Rep.', 'Pert.')
                        names = self.uq_gsa_names[file_type]
                        num_elements = [len(name_group) for name_group in names]
                        names_tmp = self.param_names
                        strw = [max(16, len(s)) for s in names_tmp]
                        nvals = len(names_tmp)
                        for i in range(nvals):
                            w = strw[i]
                            name_tmp = names_tmp[i]
                            stream_header2 += (' {:' + str(w) +'s}').format(name_tmp)
                        stream_header1 += (
                            (' {:' + str(len(stream_header2)-18) + 's}').format(
                            'Parameter values'))
                        names_tmp = [name for name_group in names for name in name_group]
                        nvals = len(names_tmp)
                        stream_header2 += (' {:16s}'*nvals).format(*names_tmp)
                        for name_idx in range(len(num_elements)):
                            stream_header1 += (' {:' +
                                str(num_elements[name_idx]*17-1) + 's}').format(
                                uq_names[name_idx])
                        f.write(stream_header1 + '\n')
                        f.write(stream_header2 + '\n')
                    for pert_no in range(npert):
                        uq_gsa_params = self.uq_gsa_params[file_type][rep_no, pert_no, :]
                        uq_gsa_data = self.uq_gsa_data[file_type][rep_no, pert_no, :]
                        data_str_fmt = '{:8d} {:8d}'
                        nvals = len(uq_gsa_params)
                        strw = [max(16, len(s)) for s in self.param_names]
                        for i in range(nvals):
                            data_str_fmt += (' {:' + str(strw[i]) +'.6E}')
                        nvals = len(uq_gsa_data)
                        data_str_fmt += (' {:16.6E}')*nvals
                        data_str = data_str_fmt.format(rep_no, pert_no,
                            *uq_gsa_params, *uq_gsa_data)
                        f.write(data_str + '\n')
        self.uq_gsa_header_written = True
