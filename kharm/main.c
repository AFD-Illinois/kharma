/******************************************************************************
 *                                                                            *
 * MAIN.C                                                                     *
 *                                                                            *
 * ESTABLISH MPI COMMUNICATION, LOOP OVER TIME, COMPLETE                      *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"
#include "defs.h"

#include <time.h>
#include <sys/stat.h>

int main(int argc, char *argv[])
{

  mpi_initialization(argc, argv);

  if (mpi_io_proc()) {
    fprintf(stdout, "\n          ************************************************************\n");
    fprintf(stdout, "          *                                                          *\n");
    fprintf(stdout, "          *                          IHARM3D                         *\n");
    fprintf(stdout, "          *                                                          *\n");
    fprintf(stdout, "          *          Ryan, Dolence & Gammie ApJ 807:31, 2015         *\n");
    fprintf(stdout, "          *                                                          *\n");
    fprintf(stdout, "          *  B R Ryan                                                *\n");
    fprintf(stdout, "          *  J C Dolence                                             *\n");
    fprintf(stdout, "          *  C F Gammie                                              *\n");
    fprintf(stdout, "          *  S M Ressler                                             *\n");
    fprintf(stdout, "          *                                                          *\n");
    fprintf(stdout, "          *                          SYNTAX                          *\n");
    fprintf(stdout, "          *                                                          *\n");
    fprintf(stdout, "          *    -p /path/to/param.dat                                 *\n");
    fprintf(stdout, "          *    -o /path/to/output/dir                                *\n");
    fprintf(stdout, "          *                                                          *\n");
    fprintf(stdout, "          ************************************************************\n\n");
  }


  // Read command line arguments

  char pfname[STRLEN] = "param.dat";
  char outputdir[STRLEN] = ".";
  for (int n = 0; n < argc; n++) {
    // Check for argv[n] of the form '-*'
    if (*argv[n] == '-' && *(argv[n]+1) != '\0' && *(argv[n]+2) == '\0' &&
        n < argc-1) {
      if (*(argv[n]+1) == 'o') { // Set output directory path
        strcpy(outputdir, argv[++n]);
      }
      if (*(argv[n]+1) == 'p') { // Set parameter file path
        strcpy(pfname, argv[++n]);
      }
    }
  }

  // Read parameter file before we move away from invocation dir
  set_core_params();
  set_problem_params();
  read_params(pfname);

  // Chdir to output directory and make output folders
  if( chdir(outputdir) != 0) {
    fprintf(stderr, "Output directory does not exist!\n");
    exit(2);
  }

  if (mpi_io_proc()) {
    int is_error = mkdir("dumps/", 0777) || mkdir("restarts/", 0777);
    if (is_error == -1 && errno != EEXIST){
      fprintf(stderr, "Could not make dumps/restarts directory.  Is output dir writeable?\n");
      exit(1);
    }
  }

  #pragma omp parallel
  {
    #pragma omp master
    {
      nthreads = omp_get_num_threads();
    }
  }

  // TODO centralize more allocations here with safe, aligned _mm_malloc
  struct GridGeom *G = calloc(1,sizeof(struct GridGeom));
  struct FluidState *S = calloc(1,sizeof(struct FluidState));

  // Perform initializations, either directly or via checkpoint
  is_restart = restart_init(G, S);
  if (!is_restart) {
    init(G, S);
    // Set globals
    nstep = 0;
    t = 0;
    dump_cnt = 0;
    // Zero the pflag array
    zero_arrays();
    if (mpi_io_proc())
      fprintf(stdout, "Initial conditions generated\n\n");
  }
  // In case we're restarting and these changed
  tdump = t + DTd;
  tlog = t + DTl;

  // Initial diagnostics
  diag(G, S, DIAG_INIT);
  if (!is_restart) restart_write(S);

  if (mpi_io_proc())
    fprintf(stdout, "t = %e tf = %e\n", t, tf);

/*******************************************************************************
    MAIN LOOP
*******************************************************************************/
  if (mpi_io_proc())
    fprintf(stdout, "\nEntering main loop\n");

  time_init();
  int dumpThisStep = 0;
  while (t < tf) {
    dumpThisStep = 0;
    timer_start(TIMER_ALL);

    // Step variables forward in time
    step(G, S);
    nstep++;

    // Don't step beyond end of run
    if (t + dt > tf) {
      dt = tf - t;
    }

    if (mpi_io_proc()) {
      fprintf(stdout, "t = %10.5g dt = %10.5g n = %8d\n", t, dt, nstep);
    }

    // File I/O with set frequencies
    if (t < tf) {
      if (t >= tdump) {
        dumpThisStep = 1;
        diag(G, S, DIAG_DUMP);
        tdump += DTd;
      }
      if (t >= tlog) {
        diag(G, S, DIAG_LOG);
        tlog += DTl;
      }
      if (nstep % DTr == 0)
        restart_write(S);
    }

    timer_stop(TIMER_ALL);

    if (nstep % DTp == 0)
      report_performance();

  }
/*******************************************************************************
    END MAIN LOOP
*******************************************************************************/
  if (dumpThisStep == 0) diag(G, S, DIAG_FINAL);

  mpi_finalize();

  return 0;
}
