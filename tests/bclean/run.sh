#!/bin/bash 
# Hyerin (02/17/23) copied from Ben's code

# Bash script testing b_clean

# User specified values here
KERR=false
bz=5e-3
DIM=3
NZONES=2 #7
BASE=8
NRUNS=2
START_RUN=0 # if this is not 0, then update start_time, out_to_in, iteration, r_out, r_in to values that you are re-starting from
DRTAG="."

# Set paths
PDR="." ## parent directory
DR="."
parfilename="./bondi_multizone_00000.par" # parameter file
KHARMA_DIR=../..

# other values determined automatically
turn_around=$(($NZONES-1))
start_time=0
out_to_in=1
iteration=1
r_out=$((${BASE}**($turn_around+2)))
r_in=$((${BASE}**$turn_around))

# if the directories are not present, make them.
mkdir -p "${DR}"
mkdir -p "${PDR}/logs/${DRTAG}"

### Start running zone by zone
for (( VAR=$START_RUN; VAR<$NRUNS; VAR++ ))
do
  args=()
  echo "${DRTAG}: iter $iteration, $VAR : t = $start_time, r_out = $r_out, r_in = $r_in"
  #logruntime=`echo "scale=20; l($r_out)*3./2-l(1.+$r_out/100000)/2." | bc -l` # round to an integer for the free-fall time (cs^2=0.01 should be updated from the desired rs value) # GIZMO
  #runtime=`echo "scale=0; e($logruntime)+1" | bc -l`
  runtime=10
  echo "Running for: " $runtime
  log_u_over_rho=-5.2915149 # test same vacuum conditions as r_shell when (rs=1e2.5)
  start_time=$(($start_time+$runtime))  

  #parfilename="../../kharma/pars/bondi_multizone/bondi_multizone_$(printf %05d ${VAR}).par" # parameter file
  
  # set problem type and cleanup
  if [ $VAR -eq 0 ]; then
    prob="bondi" #"torus" #
    init_c=1
  else
    prob="resize_restart_kharma"
    init_c=1
  fi
  
  # set BH spin
  if [[ $KERR == "true" ]]; then
    spin=0.99
  else
    spin=0.0
  fi
  
  # output time steps
  output0_dt=$((${runtime}/10))
  #output1_dt=$((${runtime}/20*10))
  output1_dt=$((${runtime}/5))
  output2_dt=$((${runtime}/10))
  
  # dt, fname, fname_fill
  if [ $VAR -ne 0 ]; then
    # update dt from the previous run
    tag=($( tail -n 10 ${PDR}/logs/${DRTAG}/log_multizone$(printf %05d $((${VAR}-1)))_out ))
    dt=$(printf "%.18g" "${tag[2]:3}") # previous dt
    dt_new=$(echo "scale=14; $dt*sqrt($BASE^(-3*$out_to_in))/4" | bc -l) # new dt ## TODO: r^3/2
    echo "dt: $dt dt_new: $dt_new"
    if (( $(echo "$dt_new > 0.00001" |bc -l) )); then
      dt_new=$dt_new
    else
      dt_new=0.00001
    fi
    fname_dir="${DR}/bondi_multizone_$(printf %05d $((${VAR}-1)))"
    echo "Restarting from directory $fname_dir"
    fname=$(find ${fname_dir} -type f -iname "*final.rhdf")
    if [ $VAR -ge $NZONES ]; then
      fname_fill_num=$((2*($iteration-1)*(${NZONES}-1)-${VAR}))
      fname_fill_dir="${DR}/bondi_multizone_$(printf %05d $fname_fill_num)"
      fname_fill=$(find ${fname_fill_dir} -type f -iname "*final.rhdf")
    else
      fname_fill="none"
    fi
    echo "Restarting with $fname, filling using $fname_fill"
    args+=(" resize_restart/fname=$fname parthenon/time/dt_min=$dt_new")
    args+=(" resize_restart/fname_fill=$fname_fill ")
  else
    r_shell=$((${r_out}/2))
    args+=(" bondi/r_shell=$r_shell ")
  fi

  # data_dir, logfiles
  data_dir="${DR}/bondi_multizone_$(printf %05d ${VAR})"
  out_fn="${PDR}/logs/${DRTAG}/log_multizone$(printf %05d ${VAR})_out"
  err_fn="${PDR}/logs/${DRTAG}/log_multizone$(printf %05d ${VAR})_err"

  $KHARMA_DIR/run.sh -n 1 -i ${parfilename} \
                      parthenon/mesh/nx1=64 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 \
                      parthenon/meshblock/nx1=64 parthenon/meshblock/nx2=64 parthenon/meshblock/nx3=64 \
                      parthenon/job/problem_id=$prob \
                      parthenon/time/tlim=${start_time} \
                      coordinates/r_in=${r_in} coordinates/r_out=${r_out}  coordinates/a=$spin coordinates/hslope=1 coordinates/transform=mks \
                      bondi/vacuum_logrho=-8.2014518 bondi/vacuum_log_u_over_rho=${log_u_over_rho} \
                      floors/disable_floors=false floors/rho_min_geom=1e-6 floors/u_min_geom=1e-8 \
                      floors/bsq_over_rho_max=100 floors/bsq_over_u_max=50 \
                      b_field/type=vertical b_field/solver=flux_ct b_field/bz=${bz} \
                      b_field/fix_flux_x1=0 b_field/initial_cleanup=$init_c \
                      b_cleanup/rel_tolerance=1.e-8 \
                      resize_restart/base=$BASE resize_restart/nzone=$NZONES resize_restart/iteration=$iteration\
                      parthenon/output0/dt=$output0_dt \
                      parthenon/output1/dt=$output1_dt \
                      parthenon/output2/dt=$output2_dt \
                      ${args[@]} \
                      -d ${data_dir} 1> ${out_fn} 2>${err_fn}
                      # kharma/b_flux_ct/seed_B_ct.cpp
                      # nlim=10000 for 1e-3   
                      # floors/u_over_rho_max=2 
                      #b_field/fix_flux_x1=1 b_field/initial_cleanup=0 \
                      #coordinates/transform=mks coordinates/hslope=1 \ this, for some reason does not work for b cleaning?

  if [ $VAR -ne 0 ]; then
    if [ $(($VAR % ($NZONES-1))) -eq 0 ]; then
      out_to_in=$(($out_to_in*(-1)))
      iteration=$(($iteration+1))
    fi
  fi

  if [ $out_to_in -gt 0 ]; then
    # half the radii
    r_out=$((${r_out}/$BASE))
    r_in=$((${r_in}/$BASE))
  else
    # double the radii
    r_out=$((${r_out}*$BASE))
    r_in=$((${r_in}*$BASE))
  fi
done
