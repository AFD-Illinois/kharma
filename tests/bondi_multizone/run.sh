#!/bin/bash 
# Hyerin (02/17/23) copied from Ben's code

# Bash script testing HD bondi

# User specified values here
KERR=false
JITTER=false #true #
ROT=false #true #
SUPEREXP=false #true #
SHORT_T_OUT=true
ROT_UPHI=0.1 # 1e-10
NZONES=8
BASE=8
NRUNS=300

# Set paths
KHARMA_DIR=$HOME/Code/kharma-multizone
PDR=$(pwd)/..
DR=$(pwd)
parfilename="bondi_multizone.par" # parameter file

turn_around=$(($NZONES-1))

# Read some values if we're restarting
if [ -f start_run ]; then
  START_RUN=$(cat start_run.var)
  start_time=$(cat start_time.var)
  out_to_in=$(cat out_to_in.var)
  iteration=$(cat iteration.var)
  r_in=$(cat r_in.var)
  r_out=$(cat r_out.var)

  last_run=$(($START_RUN - 1))
  data_dir="${DR}/$(printf %05d $last_run)"
  out_fn="${DR}/$(printf %05d $last_run).log"
else
  START_RUN=0
  start_time=0
  out_to_in=1
  iteration=1 # eq : (iteration-1)*(NZONES-1)<VAR<=iteration*(NZONES-1)
  r_in=$((${BASE}**$turn_around))
  r_out=$((${BASE}**($turn_around+2)))
  echo "STARTING MULTIZONE RUN: $START_RUN $start_time $out_to_in $iteration $r_in $r_out"
fi

# other values determined automatically
ZONE0=$((2*($NZONES-1)))
r_b=100000
log_u_over_rho=-5.2915149 # test same vacuum conditions as r_shell when (rs=1e2.5)
outermost_zone=$((2*($NZONES-1)))

# if the directories are not present, make them.
mkdir -p "${DR}"

### Start running zone by zone
for (( VAR=$START_RUN; VAR<$NRUNS; VAR++ ))
do
  # Save variables as they are when entering the loop, for restarting
  echo $VAR > start_run.var
  echo $start_time > start_time.var
  echo $out_to_in > out_to_in.var
  echo $iteration > iteration.var
  echo $r_in > r_in.var
  echo $r_out > r_out.var

  args=()
  echo "iter $iteration, $VAR : t = $start_time, r_out = $r_out, r_in = $r_in"
  logruntime=`echo "scale=20; l($r_out)*3./2-l(1.+$r_out/$r_b)/2." | bc -l` # round to an integer for the free-fall time (cs^2=0.01 should be updated from the desired rs value) # GIZMO
  runtime=`echo "scale=0; e($logruntime)+1" | bc -l`
  if [[ $SHORT_T_OUT == "true" ]]; then
    # use the next largest annulus' runtime
    if [ $(($VAR % $outermost_zone)) -eq 0 ]; then
      r_out_sm=$((${r_out}/${BASE}))
      echo "r_out=${r_out}, but I will use next largest annulus' r_out=${r_out_sm} for the runtime"
      logruntime=`echo "scale=20; l($r_out_sm)*3./2-l(1.+$r_out_sm/$r_b)/2." | bc -l`
      runtime=`echo "scale=0; e($logruntime)+1" | bc -l`
    fi
  fi
  start_time=$(($start_time+$runtime))  

  # set problem type and cleanup
  if [ $VAR -eq 0 ]; then
    prob="bondi"
  else
    prob="resize_restart_kharma"
  fi
  
  # set BH spin
  if [[ $KERR == "true" ]]; then
    spin=0.99
  else
    spin=0.0
  fi
  
  # set flow rotation
  if [[ $ROT == "true" ]]; then
    args+=(" bondi/uphi=$ROT_UPHI ")
  else
    args+=(" bondi/uphi=0 ")
  fi

  
  # output time steps
  output0_dt=$((${runtime}/100*10))
  output1_dt=$((${runtime}/20*10))
  output2_dt=$((${runtime}/1000*10))
  
  # dt, fname, fname_fill
  if [ $VAR -ne 0 ]; then
    # update dt from the previous run
    tag=($( tail -n 10 $out_fn))
    dt=$(printf "%.18g" "${tag[2]:3}") # previous dt
    dt_new=$(echo "scale=14; $dt*sqrt($BASE^(-3*$out_to_in))/4" | bc -l) # new dt ## TODO: r^3/2
    if (( $(echo "$dt_new > 0.00001" |bc -l) )); then
      dt_new=$dt_new
    else
      dt_new=0.00001
    fi
    fname=$(find $data_dir -type f -iname "*final.rhdf")
    if [ $VAR -ge $NZONES ]; then
      fname_fill_num=$((2*($iteration-1)*(${NZONES}-1)-${VAR}))
      fname_fill_dir="${DR}/$(printf %05d $fname_fill_num)"
      fname_fill=$(find ${fname_fill_dir} -type f -iname "*final.rhdf")
    else
      fname_fill="none"
    fi
    args+=(" resize_restart/fname=$fname resize_restart/use_dt=false parthenon/time/dt_min=$dt_new")
    args+=(" resize_restart/fname_fill=$fname_fill ")
  else
    r_shell=$((${r_out}/2))
    args+=(" bondi/r_shell=$r_shell ")
    if [[ $JITTER == "true" ]]; then
        args+=(" perturbation/u_jitter=0.1 ")
    else
        args+=(" perturbation/u_jitter=0.0 ")
    fi
  fi

  if [[ $SUPEREXP == "true" ]]; then
    r_br=4e6 # such that it does not lie exactly on the zone centers or faces #$((2*${BASE}**($NZONES))) # break radius (double, in order to be safe from boundaries)
    r_superexp=1e8 # outermost radius of super exponential grid

    if [ $(($VAR % $ZONE0)) -eq 0 ]; then
      args+=(" coordinates/transform=superexp ")
      args+=(" coordinates/r_br=${r_br} coordinates/npow=2.0 coordinates/cpow=1.0 ")
      #args+=(" coordinates/r_out=${r_superexp} ")
      #args+=(" coordinates/r_out=${r_out} ")
    else
      args+=(" coordinates/transform=eks ")

    fi
    args+=(" coordinates/r_out=${r_out} ")
  else
    args+=(" coordinates/transform=mks coordinates/hslope=1 coordinates/r_out=${r_out} ")
  fi
  

  # data_dir, logfiles
  data_dir="${DR}/$(printf %05d ${VAR})"
  out_fn="${DR}/$(printf %05d ${VAR}).log"
  
  ${KHARMA_DIR}/run.sh -i ${parfilename} \
                       parthenon/job/problem_id=$prob \
                       parthenon/time/tlim=${start_time} \
                       coordinates/r_in=${r_in} coordinates/a=$spin coordinates/ext_g=false \
                       bondi/vacuum_log_u_over_rho=${log_u_over_rho} bondi/use_gizmo=false \
                       b_field/type=none b_field/solver=none b_field/bz=1e-3 \
                       b_field/fix_flux_x1=false b_field/initial_cleanup=false \
                       resize_restart/base=$BASE resize_restart/nzone=$NZONES resize_restart/iteration=$iteration\
                       parthenon/output0/dt=$output0_dt \
                       parthenon/output1/dt=$output1_dt \
                       parthenon/output2/dt=$output2_dt \
                       ${args[@]} "$@" \
                       -d ${data_dir} 1>${out_fn} 2>&1

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
