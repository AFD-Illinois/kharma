#!/bin/bash 
# Hyerin (02/17/23) copied from Ben's code

# Bash script testing nonzero b flux

# User specified values here
KERR=false
JITTER=true #false #
ONEZONE=false #true # 
SHORT_T_OUT=false #true # 
ROT=false #true #
ROT_UPHI=0.1
bz=1e-4 #2e-8 #1e-8 #0 #
DIM=3 #2 # 
NZONES=3 #4 # 8 #
BASE=8
NRUNS=3000
START_RUN=494 # if !=0, change start_time, out_to_in, iteration, r_out, r_in
res=128 #64 #96 #32 #
#DRTAG="bondi_multizone_050423_bflux0_${bz}_${res}^3_n8_test_faster_rst"
DRTAG="bondi_multizone_050523_bflux0_${bz}_${res}^3_n3_noshort"
#DRTAG="bondi_multizone_050423_onezone_bflux0_${bz}_2d_n4"

# Set paths
KHARMADIR=../..
PDR="/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/" ## parent directory
DR="${PDR}data/${DRTAG}"
parfilename="${PDR}/kharma/pars/bondi_multizone/bondi_multizone_00000.par" # parameter file

# other values determined automatically
turn_around=$(($NZONES-1))
start_time=1424043 #0 #`echo "scale=20; 85184.643106985808" | bc -l` # #
out_to_in=1 #-1 #
iteration=247 #
if [[ $ONEZONE == "true" ]]; then
  r_out=$((${BASE}**($turn_around+2))) # 4096 
  r_in=1 
  NRUNS=4 #1
  #NZONES=1
else
  r_out=64 #$((${BASE}**($turn_around+2))) # 
  r_in=1 #$((${BASE}**$turn_around)) #
fi
#nlim=0

# if the directories are not present, make them.
if [ ! -d "${DR}" ]; then
  mkdir "${DR}"
fi
if [ ! -d "${PDR}logs/${DRTAG}" ]; then
  mkdir "${PDR}logs/${DRTAG}"
fi

outermost_zone=$((2*($NZONES-1)))

### Start running zone by zone
for (( VAR=$START_RUN; VAR<$NRUNS; VAR++ ))
do
  args=()
  echo "${DRTAG}: iter $iteration, $VAR : t = $start_time, r_out = $r_out, r_in = $r_in"
  if [[ $NZONES -eq 3 ]]; then
    r_b=256
  elif [[ $NZONES -eq 4 ]]; then
    r_b=256
  else
    r_b=100000
  fi
  logruntime=`echo "scale=20; l($r_out)*3./2-l(1.+$r_out/$r_b)/2." | bc -l` # round to an integer for the free-fall time (cs^2=0.01 should be updated from the desired rs value) # GIZMO
  runtime=`echo "scale=0; e($logruntime)+1" | bc -l`
  runtime=$(($runtime/2)) # test
  if [[ $SHORT_T_OUT == "true" ]]; then
    # use the next largest annulus' runtime
    if [ $(($VAR % $outermost_zone)) -eq 0 ]; then
      r_out_sm=$((${r_out}/${BASE}))
      echo "r_out=${r_out}, but I will use next largest annulus' r_out=${r_out_sm} for the runtime"
      logruntime=`echo "scale=20; l($r_out_sm)*3./2-l(1.+$r_out_sm/$r_b)/2." | bc -l`
      runtime=`echo "scale=0; e($logruntime)+1" | bc -l`
      runtime=$(($runtime/2)) # test
    fi
  fi

  #runtime=$(($runtime/4)) # test
  logrho=-8.2014518
  log_u_over_rho=-5.2915149 # test same vacuum conditions as r_shell when (rs=1e2.5)
  #if [ $VAR -ne 0 ]; then
  #  tag=($( tail -n 5 ${PDR}/logs/${DRTAG}/log_multizone$(printf %05d $((${VAR}-1)))_out ))
  #  echo ${tag[0]:5}
  #  start_time=$(printf "%.18g" "${tag[0]:5}") # last cycle number
  #  start_time=`echo "scale=0; start_time" | bc -l`
  #  echo "start time $start_time"
  #fi
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

  # set flow rotation
  if [[ $ROT == "true" ]]; then
    args+=(" bondi/uphi=$ROT_UPHI ")
  else
    args+=(" bondi/uphi=0 ")
  fi
  
  # output time steps
  #output0_dt=$((${runtime}/100*10))
  output0_dt=$((${runtime}/200*10))
  if (( $(echo "$output0_dt < 0.00001" |bc -l) )); then
    output0_dt=1
  fi
  #output1_dt=$((${runtime}/100*10))
  output1_dt=$((${runtime}/50*10))
  output2_dt=$((${runtime}/1000*10))
  
  # dt, fname, fname_fill
  if [ $VAR -ne 0 ]; then
    # update dt from the previous run
    tag=($( tail -n 10 ${PDR}/logs/${DRTAG}/log_multizone$(printf %05d $((${VAR}-1)))_out ))
    dt=$(printf "%.18g" "${tag[2]:3}") # previous dt
    #tag=($( tail -n 5 ${PDR}/logs/${DRTAG}/log_multizone$(printf %05d $((${VAR}-1)))_out ))
    #nlim=$(printf "%d" "${tag[1]:6}") # last cycle number

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
  
  # configuration
  if [[ $DIM -gt 2 ]]; then
    nx3_m=$res #32 #
    nx3_mb=$(($res/2)) #$res #
    nx2_mb=$(($res/2))
  else
    nx3_m=1
    nx3_mb=1
    nx2_mb=$res  #
  fi
  if [[ $NZONES -eq 3 ]]; then
    logrho=-4.13354231 #-5.686638255139154
    log_u_over_rho=-2.57960521 #-3.6150030239527497
    args+=(" bondi/rs=16 ")
  fi
  if [[ $NZONES -eq 4 ]]; then
    logrho=-4.200592800419657
    log_u_over_rho=-2.62430556
    args+=(" bondi/rs=16 ")
  fi
  if [[ $ONEZONE == "true" ]]; then
      #args+=(" bondi/r_shell=256 ")
      nx1_m=$((32*($NZONES+1)))
      nx1_mb=$nx1_m #128
      start_time=5100000 #10000000
      output1_dt=1000 #660 #6420 #2310
      nlim=-1
  else
      nx1_m=$res #64
      nx1_mb=$res #64
      nlim=$((100000*$res*$res/64/64)) #$(($nlim+50000)) ##$((50000*($VAR+1)))
  fi
  if [[ $bz == 0 ]]; then
    nlim=-1
  fi

  #srun --mpi=pmix ${PDR}/kharma_nlim.cuda -i ${parfilename} \
  srun --mpi=pmix ${PDR}/kharma_faster_rst_fixed.cuda -i ${parfilename} \
                                    parthenon/mesh/nx1=$nx1_m parthenon/mesh/nx2=$res parthenon/mesh/nx3=$nx3_m \
                                    parthenon/meshblock/nx1=$nx1_mb parthenon/meshblock/nx2=$nx2_mb parthenon/meshblock/nx3=$nx3_mb \
                                    parthenon/job/problem_id=$prob \
                                    parthenon/time/tlim=${start_time} \
                                    parthenon/time/nlim=$nlim \
                                    coordinates/r_in=${r_in} coordinates/r_out=${r_out} coordinates/a=$spin coordinates/ext_g=false\
                                    coordinates/transform=mks coordinates/hslope=0.3 \
                                    bondi/vacuum_logrho=${logrho} bondi/vacuum_log_u_over_rho=${log_u_over_rho} \
                                    floors/disable_floors=false floors/rho_min_geom=1e-6 floors/u_min_geom=1e-8 floors/bsq_over_u_max=1e+20 floors/adjust_k=0 \
                                    floors/bsq_over_rho_max=100 floors/u_over_rho_max=100 floors/gamma_max=10 floors/frame=drift \
                                    GRMHD/reconstruction=linear_vl GRMHD/cfl=0.5 GRMHD/add_jcon=0 \
                                    b_field/type=vertical b_field/solver=flux_ct b_field/bz=${bz} \
                                    b_field/fix_flux_x1=1 b_field/initial_cleanup=0 \
                                    resize_restart/base=$BASE resize_restart/nzone=$NZONES  resize_restart/iteration=$iteration\
                                    parthenon/output0/dt=$output0_dt \
                                    parthenon/output1/dt=$output1_dt \
                                    parthenon/output2/dt=$output2_dt \
                                    debug/flag_verbose=2 debug/extra_checks=1 \
                                    ${args[@]} \
                                    -d ${data_dir} 1> ${out_fn} 2>${err_fn}
                                    #parthenon/time/tlim=10000000 \
                                    #parthenon/output1/dt=2310 \
                                    #  b_field/type=vertical b_field/bz=${bz}flux_ct
                                    #floors/bsq_over_rho_max=100 floors/u_over_rho_max=2 \  

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
