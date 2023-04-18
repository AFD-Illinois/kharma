#!/bin/bash 
# Hyerin (02/17/23) copied from Ben's code

# Bash script testing nonzero b flux

# User specified values here
KERR=false
JITTER=true #false #
ONEZONE=true # false #
bz=1e-2 #5
DIM=3
NZONES=1 #7 #2 # 
BASE=8
NRUNS=1 #300
START_RUN=0
DRTAG="bondi_multizone_041823_bflux0_${bz}_onezone_jit"

# Set paths
KHARMADIR=../..
PDR="/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/" ## parent directory
DR="${PDR}data/${DRTAG}"
parfilename="${PDR}/kharma/pars/bondi_multizone/bondi_multizone_00000.par" # parameter file

# other values determined automatically
turn_around=$(($NZONES-1))
start_time=0 #6008119920 #
out_to_in=1 #-1 #
iteration=1 #3 #
if [[ $ONEZONE == "true" ]]; then
  r_out=4096 
  r_in=1 
  NRUNS=1
  NZONES=1
else
  r_out=$((${BASE}**($turn_around+2))) # 
  r_in=$((${BASE}**$turn_around)) #
fi

# if the directories are not present, make them.
if [ ! -d "${DR}" ]; then
  mkdir "${DR}"
fi
if [ ! -d "${PDR}logs/${DRTAG}" ]; then
  mkdir "${PDR}logs/${DRTAG}"
fi

### Start running zone by zone
for (( VAR=$START_RUN; VAR<$NRUNS; VAR++ ))
do
  args=()
  echo "${DRTAG}: iter $iteration, $VAR : t = $start_time, r_out = $r_out, r_in = $r_in"
  logruntime=`echo "scale=20; l($r_out)*3./2-l(1.+$r_out/100000)/2." | bc -l` # round to an integer for the free-fall time (cs^2=0.01 should be updated from the desired rs value) # GIZMO
  runtime=`echo "scale=0; e($logruntime)+1" | bc -l`
  runtime=$(($runtime/2)) # test
  log_u_over_rho=-5.2915149 # test same vacuum conditions as r_shell when (rs=1e2.5)
  start_time=$(($start_time+$runtime))  

  #parfilename="../../kharma/pars/bondi_multizone/bondi_multizone_$(printf %05d ${VAR}).par" # parameter file
  
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
  
  # output time steps
  output0_dt=$((${runtime}/100*10))
  output1_dt=$((${runtime}/200*10))
  #output1_dt=$((${runtime}/50*10))
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
    args+=(" resize_restart/fname=$fname parthenon/time/dt_min=$dt_new")
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

  # data_dir, logfiles
  data_dir="${DR}/bondi_multizone_$(printf %05d ${VAR})"
  out_fn="${PDR}/logs/${DRTAG}/log_multizone$(printf %05d ${VAR})_out"
  err_fn="${PDR}/logs/${DRTAG}/log_multizone$(printf %05d ${VAR})_err"
  
  if [[ $ONEZONE == "true" ]]; then
      args+=(" bondi/r_shell=256 ")
      nx1_m=128
      nx1_mb=64
      start_time=10000000
      output1_dt=2310
  else
      nx1_m=64
      nx1_mb=32
  fi

  srun --mpi=pmix ${PDR}/kharma_gcc10_bflux0.cuda -i ${parfilename} \
                                    parthenon/mesh/nx1=$nx1_m parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 \
                                    parthenon/meshblock/nx1=$nx1_mb parthenon/meshblock/nx2=64 parthenon/meshblock/nx3=32 \
                                    parthenon/job/problem_id=$prob \
                                    parthenon/time/tlim=${start_time} \
                                    coordinates/r_in=${r_in} coordinates/r_out=${r_out} coordinates/a=$spin coordinates/ext_g=false\
                                    coordinates/transform=mks coordinates/hslope=1 \
                                    bondi/vacuum_logrho=-8.2014518 bondi/vacuum_log_u_over_rho=${log_u_over_rho} \
                                    floors/disable_floors=false floors/rho_min_geom=1e-6 floors/u_min_geom=1e-8 floors/bsq_over_u_max=50 \
                                    floors/bsq_over_rho_max=100 floors/u_over_rho_max=100\
                                    GRMHD/reconstruction=linear_vl \
                                    b_field/type=vertical b_field/solver=flux_ct b_field/bz=${bz} \
                                    b_field/fix_flux_x1=1 b_field/initial_cleanup=0 \
                                    resize_restart/base=$BASE resize_restart/nzone=$NZONES  resize_restart/iteration=$iteration\
                                    parthenon/output0/dt=$output0_dt \
                                    parthenon/output1/dt=$output1_dt \
                                    parthenon/output2/dt=$output2_dt \
                                    ${args[@]} \
                                    -d ${data_dir} 1> ${out_fn} 2>${err_fn}
                                    #parthenon/time/tlim=10000000 \
                                    #parthenon/output1/dt=2310 \
                                    #  parthenon/time/nlim=$((10000*($VAR+1))) 
                                    #floors/bsq_over_rho_max=100 floors/u_over_rho_max=2 \ floors/gamma_max=5 

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