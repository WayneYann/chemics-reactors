#!/usr/bin/env python3
""" This module sets up all of the nodes in the VPU and defines an algorithm for
solving the resulting mass balances.  """

import numpy as np
import reactor_network as rn
import model_input
import model_funcs
import uq_gsa
import ode_int as ode
import io_module
import params

if model_input.use_mpi:
    try:
        from mpi4py import MPI
    except:
        # MPI not available, fall back to serial calculation
        model_input.use_mpi = False

def main():
    """
    This function is the main driver for the VPU modeling code.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Initialize MPI, if desired
    if model_input.use_mpi:
        comm = MPI.COMM_WORLD
        nproc = comm.size
        if nproc == 1:
            rank = None
        else:
            rank = comm.Get_rank()
    else:
        rank = None
        comm = None
        nproc = 1

    # Set the time clock
    t0 = model_input.t_initial
    tf = model_input.t_final
    dt = model_input.t_res
    t = t0 * 1.0

    # Initialize the network
    network = rn.setup_network(model_input.nodes, model_input.streams,
        model_funcs.rate_law, model_funcs.get_mass_flow_rates)

    # Initialize the system of ODEs/DAEs
    solver = model_input.solver
    settings = model_input.solver_settings
    ode_sys = ode.ode_integrator(solver, settings, network)
    ode_sys.set_ic(t0, reset=False)
    ode_sys.set_user_params()

    # Initialize the output files
    if model_input.uq_gsa_settings['type'] is not None:
        output_files = io_module.io_handler(
            base_files=model_input.output_filenames,
            uq_gsa_files=model_input.uq_gsa_settings['output_filenames'],
            uq_gsa_output=model_input.uq_gsa_settings['output'], rank=rank)
    else:
        output_files = io_module.io_handler(
            base_files=model_input.output_filenames, rank=rank)

    if rank is None or rank == 0:
        # Step through time
        while t <= tf:

            # Integrate one time step from t to t + dt
            ode_sys.integrate(t, t+dt)
#            print('Time: ' + str(t) + ' s')

            # Write basic transient data for debugging and convergence testing
            output_files.write_y(t, ode_sys.network, ode.vode_y)

            # TODO: implement some basic error control to restart integration if
            # too much work is done or bail out if an unrecoverable error
            # occurred.

            # TODO: can this be with Python's warnings module by sending it to a
            # string?

            # It would be better to use VODE's native return codes, but SciPy
            # doesn't expose those without a patch to the source code...

            # Update time
            t += dt

            # Something went wrong; abort the integration
            if not ode_sys.ode_obj.successful():
                break

        # Write output
        output_files.get_data(ode_sys.network)
        output_files.write_output(t)

    if model_input.uq_gsa_settings['type'] is not None:
        # Initialize UQ/GSA trajectory sequences and data structures
        if rank is not None:
            uq_gsa_traj = uq_gsa.uq_gsa_seq(params.param_dists,
                design_type=model_input.uq_gsa_settings['type'],
                seed=model_input.uq_gsa_settings['rand_seed'][rank],
                doe_design=model_input.uq_gsa_settings['DOE_design'])
        else:
            uq_gsa_traj = uq_gsa.uq_gsa_seq(params.param_dists,
                design_type=model_input.uq_gsa_settings['type'],
                seed=model_input.uq_gsa_settings['rand_seed'][0],
                doe_design=model_input.uq_gsa_settings['DOE_design'])

        # Initialize memory for UQ/GSA data
        nrep = model_input.uq_gsa_settings['replicates']
        nvec = uq_gsa_traj.max_iter
        output_files.init_uq_gsa_data(ode_sys.network, uq_gsa_traj, nrep, nvec)

        # Loop over all points in the trajectory
        for n in range(nrep):

            print(n)
            # This is here to keep worker processes from solving a base DOE
            # model
            if rank is None:
                rep_idx = n
            else:
                # The only way this will be zero is if both n and rank are zero
                rep_idx = max(n, rank)

            # Get data for this block of points
            uq_gsa_traj.generate_vectors(rep_idx)
            for m in range(nvec):

                uq_gsa_traj.perturb_params(params)
                ode_sys.set_ic(t0, reset=True)
                t = 0.0
                failure = False
                while t <= tf:
                    ode_sys.integrate(t, t+dt)
                    t += dt
                    if not ode_sys.ode_obj.successful():
                        failure = True
                        break
                output_files.set_uq_gsa_data(ode_sys.network, uq_gsa_traj, n, m,
                    failure=failure)

            # Write the output for this replicate
            output_files.write_uq_gsa_output(t, n, nvec)

        # Save final versions of the raw response data
        output_files.save_uq_gsa_data(rank)

if __name__ == "__main__":
    main()
