#!/usr/bin/env python

# This runs a "multi-zone" KHARMA sequence
# See --help

import os
import sys
import click
import glob
import subprocess
import pickle

import numpy as np
import h5py

import pyharm


def format_args(args):
    """Format a dict in var=val format for Parthenon"""
    arg_list = []
    for key in args.keys():
        arg_list += [key + "={}".format(args[key]).lower()]
    return arg_list


def calc_runtime(r_out, r_b, loctchar):
    """r/v where v=sqrt(v_ff**2+c_s**2)"""
    if loctchar:
        return r_out / np.sqrt(1.0 / r_out + 1.0 / r_b)
    else:
        return np.power(min(r_out, r_b), 3.0 / 2)


def data_dir(n):
    """Data directory naming scheme"""
    return "{:05d}".format(n)


def calc_nx1(kwargs, r_out=None, r_in=None):  # (given_nx1, nzones):
    """adjust to a new nx1 for a larger annulus to have effectively the same resolution as other annuli"""
    if r_out is None:
        r_out = kwargs["r_out"]
    if r_in is None:
        r_in = kwargs["r_in"]
    nzones_plus_one = int(np.log(r_out / r_in) / np.log(kwargs["base"]))  # equal to (nzones+1)
    given_nx1 = kwargs["nx1"]
    nx1 = int((given_nx1 / 2.0) * (nzones_plus_one))
    return nx1


def calc_rb(kwargs):
    gam = kwargs["gamma"]
    n = 1.0 / (gam - 1)
    rs = float(kwargs["rs"])
    if abs(gam - 5.0 / 3.0) < 1e-2:
        # only when gamma = 5/3, rb ~ rs^2
        if kwargs["incorrectrb"]:
            r_b = rs**2
        else:
            r_b = 80.0 * rs**2 / (27.0 * gam)
    else:
        # otherwise, rb ~ rs
        r_b = 4 * (1 + n) * rs / ((2 * (n + 3) - 9) * gam)
    return r_b


@click.command()
# Run parameters
@click.option("--nx1", default=64, help="1-Run radial resolution")
@click.option("--nx2", default=64, help="1-Run theta resolution")
@click.option("--nx3", default=64, help="1-Run phi resolution")
@click.option("--nx1_mb", default=64, help="1-Run radial block resolution")
@click.option("--nx2_mb", default=32, help="1-Run theta block resolution")
@click.option("--nx3_mb", default=32, help="1-Run phi block resolution")
@click.option("--nzones", default=8, help="Total number of zones (annuli)")
@click.option("--base", default=8.0, type=float, help="Exponent base for annulus sizes")
@click.option("--nruns", default=3000, help="Total number of runs to perform")
@click.option("--spin", default=0.0, help="BH spin")
@click.option("--bz", default=0.0, help="B field Z component. Zero for no field")
@click.option("--cfl", default=0.9, help="Courant condition fraction.  Defaults to 0.5 in B field")
@click.option("--tlim", default=None, help="Enforce a specific tlim for every run (for testing)")
@click.option("--tmax", default=500., help="Maximum time in units of Bondi time")
@click.option("--nlim", default=float(5e4), help="Consistent max number of steps for each run")
@click.option("--rs", default=np.sqrt(1.0e5), help="sonic radius. None chooses based on nzones")
@click.option("--mdot", default=1.0, help="mdot.")
@click.option("--jitter", default=0.0, help="Proportional jitter to apply to starting state. Default 10% w/B field")
# Flags and options
@click.option("--kharma_bin", default="kharma.cuda", help="Name (not path) of KHARMA binary to run")
@click.option("--kharma_args", default="", help="Arguments for KHARMA run.sh")
@click.option("--long_t_in", is_flag=True, help="Use longer time for innermost annulus")
@click.option("--restart", is_flag=True, help="Restart from most recent run parameters")
@click.option("--parfile", default=None, help="Parameter filename")
@click.option("--gizmo", is_flag=True, help="Start from GIZMO data")
@click.option("--gizmo_fname", default="../gizmo_data.txt", help="Filename of GIZMO data")
@click.option("--ext_g", is_flag=True, help="Include external gravity")
# Added by Hyerin
@click.option("--onezone", is_flag=True, help="Run onezone instead.")
@click.option("--recon", default=None, help="reconstruction method.")
@click.option("--combine_out_ann", is_flag=True, help="Combine outer annuli larger than Bondi radius.")
@click.option("--combine_in_ann", is_flag=True, help="Combine two innermost annuli.")
@click.option("--move_rin", is_flag=True, help="Move r_in instead of switching btw same sized annuli.")
@click.option("--gamma_max", default=10, help="Gamma_max floor.")
@click.option("--sigma_max", default=100, help="bsq_over_rho_max floor.")
@click.option("--b2u_max", default=None, help="bsq_over_u_max floor.")
@click.option("--gamma", default=5.0 / 3, help="adiabatic index.")
@click.option("--rhomin", default=1e-6, help="rho min geom.")
@click.option("--umin", default=1e-8, help="u min geom.")
@click.option("--btype", default="r1s2", help="b field type")
@click.option("--coord", default=None, help="coordinate system")
@click.option("--smoothness", default=0.02, help="smoothness for WKS")
@click.option("--frame", default="drift", help="Specify a frame when applying floors.")
@click.option("--urfrac", default=0.0, help="ur_frac")
@click.option("--uphi", default=0.0, help="uphi")
@click.option("--incorrectrb", is_flag=True, help="Use previous incorrect R_B calculation instead.")
@click.option("--one_trun", is_flag=True, help="For innermost and outermost annuli, run for 1 t_char instead of 2.")
@click.option("--loctchar", is_flag=True, help="Use local charateristic time instead of capping it at the Bondi time.")
@click.option("--bclean", is_flag=True, help="Clean divergence of B fields.")
@click.option("--b_ct", is_flag=True, help="Use face-centered B fields instead of cell-centered.")
@click.option("--derefine_poles", is_flag=True, help="Derefine poles for internal SMR.")
@click.option("--derefine_nlevels", default=1, help="Derefine number of levels for internal SMR.")
@click.option("--output0_dt", default=None, help="output0 dt.")
@click.option("--dont_kill_on_divb", is_flag=True, help="Don't kill the simulation when the divB is too large. Mostly for test purposes.")
# Don't use this
@click.option("--start_time", default=0.0, help="Starting time. Only use if you know what you're doing.")
def run_multizone(**kwargs):
    """This script runs a "multi-zone" KHARMA sequence.
    The idea is to divide a large domain (~1e8M radius) into several "zones,"
    then evolve them one at a time while keeping the others constant.
    This allows recovering long-term steady-state behavior quickly, by evolving each
    "zone" on its own timescale.
    Each run takes the final state of the last run, expands the domain inward or outward, and
    evolves the resulting domain/state.

    This mode now supports magnetic fields, arbitrary overlaps and coordinates, and other niceties.
    """
    # We're kept in a script subdirectory in kharma/
    mz_dir = os.path.dirname(os.path.realpath(__file__))
    # parent
    kharma_dir = mz_dir + "/../.."
    # Get our name from the working dir
    run_name = os.getcwd().split("/")[-1]

    # Assign initial arguments, based on either:
    # 1. Loading last-started run when restarting
    # 2. Computing arguments from kwargs if beginning fresh
    if kwargs["restart"]:
        kwargs, args = copy_from_restart(kwargs)
    else:
        # First run arguments
        base = kwargs["base"]
        args = {}
        args["parthenon/job/problem_id"] = "bondi"
        args["resize_restart/base"] = base
        args["resize_restart/nzone"] = kwargs["nzones"]
        args["resize_restart/iteration"] = 1
        kwargs["start_run"] = 0

        turn_around = kwargs["nzones"] - 1
        args["coordinates/r_out"] = base ** (turn_around + 2)
        if kwargs["onezone"]:
            args["coordinates/r_in"] = 1
        else:
            args["coordinates/r_in"] = base**turn_around
        # Initialize half-vacuum, unless it's the first GIZMO run
        # NOT USING GIZMO, so not using r_shell for now
        #if kwargs["gizmo"]:
        #    args["bondi/r_shell"] = 3e6  # args['coordinates/r_in'] # not using r_shell
        #else:
        #    args["bondi/r_shell"] = base ** (turn_around + 2) / 2.0

        # bondi & vacuum parameters
        # TODO derive these from r_b or gizmo
        #if args["coordinates/r_out"] < 1e5 and kwargs["bz"] > 1e-4:  # kwargs['nzones'] == 3 or kwargs['nzones'] == 6:
        #    kwargs["rs"] = 16
        #    logrho = -4.13354231
        #    log_u_over_rho = -2.57960521
        #elif kwargs["nzones"] == 4:
        #    kwargs["rs"] = 16
        #    logrho = -4.13354231
        #    logrho = -4.200592800419657
        #    log_u_over_rho = -2.62430556
        if kwargs["gizmo"]:
            # kwargs['r_b'] = 1e5
            logrho = -8.33399171  # -7.80243572
            log_u_over_rho = -5.34068635
        else:
            # kwargs['r_b'] = 1e5
            logrho = 0  # -8.2014518
            log_u_over_rho = -5.2915149
        args["bondi/mdot"] = kwargs["mdot"]
        args["bondi/vacuum_logrho"] = logrho
        args["bondi/vacuum_log_u_over_rho"] = log_u_over_rho
        args["bondi/rs"] = kwargs["rs"]
        args["bondi/r_b"] = calc_rb(kwargs)
        args["bondi/ur_frac"] = kwargs["urfrac"]
        args["bondi/uphi"] = kwargs["uphi"]

        # B field additions
        if kwargs["bz"] != 0.0:
            # Set a field to initialize with
            args["b_field/type"] = kwargs["btype"]  # "r1s2" #"vertical"
            if (kwargs["b_ct"]): args["b_field/solver"] = "face_ct"
            else: args["b_field/solver"] = "flux_ct"
            args["b_field/A0"] = kwargs["bz"]
            if kwargs["bclean"]:
                args["b_field/initial_cleanup"] = 1
                args["b_cleanup/rel_tolerance"] = 1.e-5
            if (kwargs["dont_kill_on_divb"]): args["b_field/kill_on_large_divb"] = 0
            # Compress coordinates to save time
            if kwargs["nx2"] >= 128 and not kwargs["onezone"]:
                args["coordinates/transform"] = "fmks"
                args["coordinates/mks_smooth"] = 0.0
                args["coordinates/poly_xt"] = 0.8
                args["coordinates/poly_alpha"] = 16
            else:
                args["coordinates/transform"] = "mks"
                args["coordinates/hslope"] = 0.3
            # Enable the floors
            args["floors/disable_floors"] = False
            args["floors/gamma_max"] = kwargs["gamma_max"]
            args["floors/bsq_over_rho_max"] = kwargs["sigma_max"]
            if kwargs["b2u_max"] is not None: args["floors/bsq_over_u_max"] = float(kwargs["b2u_max"])
            args["floors/frame"] = kwargs["frame"]
            # And modify a bunch of defaults
            # Assume we will always want jitter if we have B unless a 2D problem
            if kwargs["nx3"] > 1:  #
                kwargs["jitter"] = 0.1
            # Lower the cfl condition in B field
            kwargs["cfl"] = 0.5
            if kwargs["recon"] is None:
                # use weno5
                args["driver/reconstruction"] = "weno5"
        if kwargs["coord"] is not None:
            args["coordinates/transform"] = kwargs["coord"]
            if kwargs["coord"] == "fmks":
                # only for fmks
                args["coordinates/mks_smooth"] = 0.0
                args["coordinates/poly_xt"] = 0.8
                args["coordinates/poly_alpha"] = 14
            elif kwargs["coord"] == "wks":
                # TODO these are only for wks
                args["coordinates/lin_frac"] = 0.6  # 0.75
                args["coordinates/smoothness"] = kwargs["smoothness"]
        if kwargs["recon"] is not None:
            if 'lower_edges' in kwargs["recon"]: args["driver/lower_edges"] = 1
            if 'lower_poles' in kwargs["recon"]: args["driver/lower_poles"] = 1
            if 'lower' in kwargs["recon"]: kwargs["recon"] = "weno5"
            args["driver/reconstruction"] = kwargs["recon"]
        args["GRMHD/gamma"] = kwargs["gamma"]
        if (kwargs["derefine_poles"]): 
            args["GRMHD/ismr_poles"] = 1
            args["GRMHD/ismr_nlevels"] = kwargs["derefine_nlevels"]
        args["floors/rho_min_geom"] = kwargs["rhomin"]
        args["floors/u_min_geom"] = kwargs["umin"]

        # Parameters directly from defaults/cmd
        args["perturbation/u_jitter"] = kwargs["jitter"]
        args["GRMHD/cfl"] = kwargs["cfl"]
        args["coordinates/a"] = kwargs["spin"]
        args["coordinates/ext_g"] = kwargs["ext_g"]
        args["bondi/use_gizmo"] = kwargs["gizmo"]
        args["gizmo_shell/datfn"] = kwargs["gizmo_fname"]
        args["parthenon/time/nlim"] = kwargs["nlim"]

        # effective nzones (Hyerin 07/27/23)
        if (kwargs["combine_out_ann"]) and not kwargs["onezone"]:
            # think what's the smallest annulus where the logarithmic middle radius is larger than r_b
            # (i.e. 8^n > 1e5 for base=8 r_b=1e5 where n is the nth smallest annulus)
            kwargs["nzones_eff"] = int(np.ceil(np.log(args["bondi/r_b"]) / np.log(kwargs["base"])))
            args["coordinates/r_in"] = base ** (kwargs["nzones_eff"] - 1)
            if kwargs["base"] < 2:  # this means that the second smallest annulu's r_in is inside the horizon
                args["coordinates/r_in"] = base ** (kwargs["nzones_eff"])
        elif (kwargs["combine_in_ann"]) and not kwargs["onezone"]:
            kwargs["nzones_eff"] = kwargs["nzones"] - 1
        else:
            kwargs["nzones_eff"] = kwargs["nzones"]
        args["resize_restart/nzone_eff"] = kwargs["nzones_eff"]

        # Mesh size
        args["parthenon/mesh/nx1"] = calc_nx1(kwargs, args["coordinates/r_out"], args["coordinates/r_in"])
        args["parthenon/mesh/nx2"] = kwargs["nx2"]
        args["parthenon/mesh/nx3"] = kwargs["nx3"]
        args["parthenon/meshblock/nx1"] = kwargs["nx1_mb"] / kwargs["nx1"] * args["parthenon/mesh/nx1"]
        args["parthenon/meshblock/nx2"] = kwargs["nx2_mb"]
        args["parthenon/meshblock/nx3"] = kwargs["nx3_mb"]

    # Any derived parameters once we've loaded args/kwargs
    # Default parameters are in mz_dir
    if kwargs["parfile"] is None:
        kwargs["parfile"] = mz_dir + "/multizone.par"

    stop = False
    r_b = args["bondi/r_b"]
    base = args["resize_restart/base"]
    # Iterate, starting with the default args and updating as we go
    for run_num in np.arange(kwargs["start_run"], kwargs["nruns"]):
        # run times for each annulus
        r_out = args["coordinates/r_out"]
        if kwargs["tlim"] is None:
            # Calculate characteristic time
            if not kwargs["move_rin"]:
                runtime = calc_runtime(r_out, r_b, kwargs["loctchar"])
            else:
                r_out_temp = args["coordinates/r_in"] * base ** 2
                if kwargs["combine_in_ann"] and args["coordinates/r_in"] <= base:
                    r_out_temp = base ** 3
                runtime = calc_runtime(r_out_temp, r_b, kwargs["loctchar"])
            # B field runs use half this
            if kwargs["bz"] != 0.0:
                runtime /= np.power(base, 3.0 / 2) * 2  # half of free-fall time at the log middle radius
            if args["coordinates/r_in"] >= base ** (kwargs["nzones"] - 1) and (not kwargs['one_trun']):
                # double the runtime for the outermost annulus
                runtime *= 2
            if args["coordinates/r_in"] < 2:
                if (not kwargs['one_trun']): runtime *= 2  # double the runtime for innermost annulus
                if kwargs["long_t_in"]:
                    print("LONG_T_IN @ RUN # {}: using longer runtime".format(run_num))
                    runtime *= 5  # 5 tff at the log middle radius
                    if kwargs['one_trun']: runtime *= 2 # compensate one_trun here
        else:
            runtime = float(kwargs["tlim"])
        if args["coordinates/r_in"] < 2:
            args["boundaries/inner_x1"] = "outflow"
            args["boundaries/check_inflow_inner_x1"] = 1
        else:
            args["boundaries/inner_x1"] = "dirichlet"
            args["boundaries/check_inflow_inner_x1"] = 0

        tlim = kwargs["start_time"] + runtime
        if kwargs["onezone"]:
            tlim = runtime
        tlim_max = float(kwargs["tmax"]) * calc_runtime(kwargs["base"] ** (kwargs["nzones"] + 1), r_b, kwargs["loctchar"]) #np.power(r_b, 3.0 / 2.0)
        print(tlim_max)
        if tlim > tlim_max:
            stop = True
        args["parthenon/time/tlim"] = tlim

        # Output timing (TODO make options)
        if kwargs["onezone"]:
            runtime = calc_runtime(r_out, r_b, False) # 1-zone will only be run with smaller Bondi problem where trun is capped.
        if kwargs["output0_dt"] is None: args["parthenon/output0/dt"] = max((runtime / 10.0), 1e-7)
        else: args["parthenon/output0/dt"] = float(kwargs["output0_dt"])
        args["parthenon/output1/dt"] = max((runtime / 5.0), 1e-7)  #
        args["parthenon/output2/dt"] = runtime / 10  # 0.

        # Start any future run from this point
        kwargs["start_run"] = run_num

        # Now that we've determined all parameters, save them as used
        restart_file = open("restart.p", "wb")
        pickle.dump(kwargs, restart_file)
        pickle.dump(args, restart_file)
        restart_file.close()
        # And print them
        print(run_name + ": iter {}, run {} : radius {:.4g} to {:.4g}, time {:.4g} to {:.4g}".format(args["resize_restart/iteration"], run_num, args["coordinates/r_in"], args["coordinates/r_out"], kwargs["start_time"], args["parthenon/time/tlim"]))

        ddir = data_dir(run_num)
        os.makedirs(ddir, exist_ok=True)
        fout = open(ddir + "/kharma.log", "w")
        if kwargs["kharma_bin"] not in ["", "kharma.cuda"]:
            kharma_bin_arg = ["-b", kwargs["kharma_bin"]]
        else:
            kharma_bin_arg = []
        ret_obj = subprocess.run([kharma_dir + "/run.sh"] + kharma_bin_arg + ["-i", kwargs["parfile"], "-d", ddir] + format_args(args), stdout=fout, stderr=subprocess.STDOUT)
        fout.close()

        # Don't continue (& save restart data, etc) if KHARMA returned error
        if ret_obj.returncode != 0:
            print("KHARMA returned error: {}. Exiting.".format(ret_obj.returncode))
            exit(-1)
        if stop:
            print("tlim max reached!")
            break

        # Update parameters for the next pass
        # This updates both kwargs (start_time) and args (coordinates, dt, iteration #, fnames)
        update_args(run_num, kwargs, args)

def copy_from_restart(kwargs):
    # Crude, but I need to know what was passed to override on restore
    kwargs_save = {}
    for arg in [a.replace("-", "").split("=")[0] for a in sys.argv[1:] if "-" in a]:
        kwargs_save[arg] = kwargs[arg]
    restart_file = open("restart.p", "rb")
    kwargs = {**kwargs, **pickle.load(restart_file)}
    args = pickle.load(restart_file)
    restart_file.close()
    if kwargs["onezone"]:
        # if onezone, just inherit everything from restart.p except nruns
        kwargs["nruns"] = kwargs_save["nruns"]
        update_args(kwargs["start_run"], kwargs, args)
        kwargs["start_run"] += 1
    for arg in kwargs_save.keys():
        if "nlim" in arg or "kharma_bin" in arg or "tmax" in arg:  # can change nlim from previous run
            kwargs[arg] = kwargs_save[arg]
    args["parthenon/time/nlim"] = kwargs["nlim"]
    if 'bondi/r_b' not in args.keys():
        args['bondi/r_b'] = calc_rb(kwargs)

    return kwargs, args
def update_args(run_num, kwargs, args):
    # Update the dictionary of args to prepare for the *next* run (run_num+1).
    # Called after the first run is finished, and after each run subsequently

    # We'll always be restarting after the first run
    args["parthenon/job/problem_id"] = "resize_restart_kharma"

    # Filename to restart from
    fname_dir = data_dir(run_num)
    if kwargs["onezone"]:
        fname = sorted(glob.glob(fname_dir + "/*.rhdf"))[-1]
    else:
        fname = glob.glob(fname_dir + "/*final.rhdf")[0]
    # Get start_time, ncycle, dt from previous run
    kwargs["start_time"] = pyharm.io.get_dump_time(fname)
    args["parthenon/time/ncycle"] = 0
    d = pyharm.load_dump(fname)
    iteration = d["iteration"]
    last_r_out = d["r_out"]
    last_r_in = d["r_in"]
    del d
    # TODO read all of Params/Info in pyharm
    f = h5py.File(fname, "r")
    dt_last = f["Params"].attrs["Globals/dt_last"]
    f.close()

    # Increment iteration count when we just finished the outermost zone
    if run_num > 0 and run_num % (kwargs["nzones_eff"] - 1) == 0:
        iteration += 1
    args["resize_restart/iteration"] = iteration

    # Are we moving inward?
    out_to_in = (-1) ** (1 + iteration)  # if iteration odd, out_to_in=1, if even, out_to_in=-1

    # Choose timestep and radii for the next run: smaller/larger as we step in/out
    if not kwargs["onezone"]:
        args["parthenon/time/dt"] = max(dt_last * kwargs["base"] ** (-3.0 / 2.0 * out_to_in) / 4, 1e-5)
        if out_to_in > 0:
            if not kwargs["move_rin"]:
                args["coordinates/r_out"] = last_r_in * kwargs["base"]  # last_r_out / kwargs['base']
            args["coordinates/r_in"] = last_r_in / kwargs["base"]
        else:
            if not kwargs["move_rin"]:
                args["coordinates/r_out"] = last_r_out * kwargs["base"]
            args["coordinates/r_in"] = last_r_in * kwargs["base"]

        if (kwargs["combine_out_ann"]) and args["coordinates/r_in"] >= kwargs["base"] ** (kwargs["nzones_eff"] - (kwargs["base"] > 2)):
            # if the next simulation is at the largest annulus,
            # make r_out and nx1 larger
            # if base < 2, the largest r_in is base^nzones_eff. if not, base^nzones_eff-1
            args["coordinates/r_out"] = kwargs["base"] ** (kwargs["nzones"] + 1)
        if (kwargs["combine_in_ann"]) and args["coordinates/r_in"] <= kwargs["base"]:
            if out_to_in > 0 : args["coordinates/r_in"] = 1
            else: args["coordinates/r_in"] *= kwargs["base"]
        args["parthenon/mesh/nx1"] = calc_nx1(kwargs, args["coordinates/r_out"], args["coordinates/r_in"])
        args["parthenon/meshblock/nx1"] = args["parthenon/mesh/nx1"]

    # Get filename to fill in the rest that fname doesn't cover
    if run_num + 1 < kwargs["nzones_eff"]:
        fname_fill = "none"
    else:
        # TODO explain why this number is correct
        fname_fill_dir = data_dir(2 * (iteration - 1) * (kwargs["nzones_eff"] - 1) - (run_num + 1))
        fname_fill = glob.glob(fname_fill_dir + "/*final.rhdf")[0]
        args["perturbation/u_jitter"] = 0.0  # jitter is turned off when not initializing.
    args["resize_restart/fname"] = fname
    args["resize_restart/fname_fill"] = fname_fill


if __name__ == "__main__":
    run_multizone()
