#!/bin/bash 
# Hyerin (02/18/23) copied from Ben's code

# Bash script testing gizmo shell run (no b field)

# User specified values here
KERR=false
EXT_G=false #true #
SHORT_T_OUT=true #false
DIM=3
NZONES=8 #7
BASE=8
NRUNS=300
START_RUN=0 # if this is not 0, then update start_time, out_to_in, iteration, r_out, r_in to values that you are re-starting from
DRTAG="bondi_multizone_052523_gizmo_n8_64^3_noshock"

# Set paths
KHARMADIR=../..
PDR="/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/" ## parent directory
DR="${PDR}data/${DRTAG}"
parfilename="${PDR}/kharma/pars/bondi_multizone/bondi_multizone_00000.par" # parameter file

# other values determined automatically
turn_around=$(($NZONES-1))
start_time=0 #19493738388 #
out_to_in=1 #-1 # 
iteration=1 #6 # eq : (iteration-1)*(NZONES-1)<VAR<=iteration*(NZONES-1)
r_out=$((${BASE}**($turn_around+2))) #512 #
r_in=$((${BASE}**$turn_around)) #8 #

# if the directories are not present, make them.
if [ ! -d "${DR}" ]; then
  mkdir "${DR}"
fi
if [ ! -d "${PDR}logs/${DRTAG}" ]; then
  mkdir "${PDR}logs/${DRTAG}"
fi

outermost_zone=$((2*($NZONES-1)))
r_b=100000

# GIZMO data
#datfn="${PDR}/data/gizmo/042623_3d/new_snap_deposit_010.hdf5"
datfn="${PDR}/data/gizmo/031623_100Myr/dat.txt"
logrho=-7.80243572
log_u_over_rho=0 #-5.34068635

### Start running zone by zone
for (( VAR=$START_RUN; VAR<$NRUNS; VAR++ ))
do
  args=()
  echo "${DRTAG} iter $iteration, $VAR : t = $start_time, r_out = $r_out, r_in = $r_in"
  logruntime=`echo "scale=20; l($r_out)*3./2-l(1.+$r_out/$r_b)/2." | bc -l` # round to an integer for the free-fall time (cs^2=0.01 should be updated from the desired rs value) # GIZMO
  if [[ $SHORT_T_OUT == "true" ]]; then
    # use the next largest annulus' runtime
    if [ $(($VAR % $outermost_zone)) -eq 0 ]; then
      r_out_sm=$((${r_out}/${BASE}))
      echo "r_out=${r_out}, but I will use next largest annulus' r_out=${r_out_sm} for the runtime"
      logruntime=`echo "scale=20; l($r_out_sm)*3./2-l(1.+$r_out_sm/$r_b)/2." | bc -l`
    fi
  fi
  runtime=`echo "scale=0; e($logruntime)+1" | bc -l`
  #log_u_over_rho=-5.2915149 # test same vacuum conditions as r_shell when (rs=1e2.5)
  start_time=$(($start_time+$runtime))  
  
  # set problem type and cleanup
  if [ $VAR -eq 0 ]; then
    prob="bondi" #"gizmo_shell"
  else
    prob="resize_restart_kharma"
  fi
  
  # set BH spin
  if [[ $KERR == "true" ]]; then
    spin=0.99
  else
    spin=0.0
  fi
  
  # output time steps
  output0_dt=$((${runtime}/100*10))
  output1_dt=$((${runtime}/20*10))
  output2_dt=$((${runtime}/1000*10))
  
  # dt, fname, fname_fill
  if [ $VAR -ne 0 ]; then
    # update dt from the previous run
    tag=($( tail -n 10 ${PDR}/logs/${DRTAG}/log_multizone$(printf %05d $((${VAR}-1)))_out ))
    dt=$(printf "%.18g" "${tag[2]:3}") # previous dt
    dt_new=$(echo "scale=14; $dt*sqrt($BASE^(-3*$out_to_in))/4" | bc -l) # new dt ## TODO: r^3/2
    if (( $(echo "$dt_new > 0.00001" |bc -l) )); then
      dt_new=$dt_new
    else
      dt_new=0.00001
    fi
    fname_dir="${DR}/bondi_multizone_$(printf %05d $((${VAR}-1)))"
    fname=$(find ${fname_dir} -type f -iname "*final.rhdf")
    if [ $VAR -ge $NZONES ]; then
      fname_fill_num=$((2*($iteration-1)*(${NZONES}-1)-${VAR}))
      fname_fill_dir="${DR}/bondi_multizone_$(printf %05d $fname_fill_num)"
      fname_fill=$(find ${fname_fill_dir} -type f -iname "*final.rhdf")
    else
      fname_fill="none"
    fi
    args+=(" resize_restart/fname=$fname resize_restart/use_dt=false parthenon/time/dt_min=$dt_new")
    args+=(" resize_restart/fname_fill=$fname_fill ")
  else
    r_shell=$r_in
    args+=(" bondi/r_shell=$r_shell ")
  fi

  # data_dir, logfiles
  data_dir="${DR}/bondi_multizone_$(printf %05d ${VAR})"
  out_fn="${PDR}/logs/${DRTAG}/log_multizone$(printf %05d ${VAR})_out"
  err_fn="${PDR}/logs/${DRTAG}/log_multizone$(printf %05d ${VAR})_err"

  #srun --mpi=pmix ${PDR}/kharma_fork/kharma.cuda -i ${parfilename}  \
  srun --mpi=pmix ${PDR}/kharma_faster_rst_fixed.cuda -i ${parfilename}  \
                                    parthenon/job/problem_id=$prob \
                                    parthenon/mesh/nx1=64 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 \
                                    parthenon/meshblock/nx1=64 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=32 \
                                    parthenon/time/tlim=${start_time} \
                                    coordinates/r_in=${r_in} coordinates/r_out=${r_out} coordinates/a=$spin coordinates/ext_g=$EXT_G \
                                    coordinates/transform=mks coordinates/hslope=1 \
                                    GRMHD/reconstruction=linear_vl \
                                    bondi/vacuum_logrho=${logrho} bondi/vacuum_log_u_over_rho=${log_u_over_rho} \
                                    bondi/use_gizmo=true gizmo_shell/datfn=${datfn} \
                                    b_field/type=none b_field/solver=none \
                                    b_field/fix_flux_x1=0 b_field/initial_cleanup=0 \
                                    resize_restart/base=$BASE resize_restart/nzone=$NZONES resize_restart/iteration=$iteration \
                                    parthenon/output0/dt=$output0_dt \
                                    parthenon/output1/dt=$output1_dt \
                                    parthenon/output2/dt=$output2_dt \
                                    ${args[@]} \
                                    -d ${data_dir} 1> ${out_fn} 2>${err_fn}
                                    # logrho=-5.34068635
                                    #gizmo_shell/datfn="/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/data/gizmo/041423_3d/new_snap_deposit_020.hdf5" 

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
