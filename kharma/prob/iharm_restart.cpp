
#include "iharm_restart.hpp"

#include "hdf5_utils.h"
#include "mpi.hpp"

#include <sys/stat.h>
#include <ctype.h>

// First boundary sync
void outflow_x1(const GRCoordinates& G, GridVars P, int n1, int n2, int n3);
void polar_x2(const GRCoordinates& G, GridVars P, int n1, int n2, int n3);
void periodic_x3(const GRCoordinates& G, GridVars P, int n1, int n2, int n3);

using namespace Kokkos;

// TODO
// At least check that Rin,Rout match
// Actually look at Rin,Rout,gamma and (re)build the Coordinates and mesh on them
// Re-gridding algorithm
// Start with multiple meshes i.e. find full file dimensions, where to start reading

double ReadIharmRestart(MeshBlock *pmb, GRCoordinates G, GridVars P, std::string fname)
{
    IndexDomain domain = IndexDomain::interior;
    // Full mesh size
    hsize_t n1tot = pmb->pmy_mesh->mesh_size.nx1;
    hsize_t n2tot = pmb->pmy_mesh->mesh_size.nx2;
    hsize_t n3tot = pmb->pmy_mesh->mesh_size.nx3;
    // Our block size, start, and bounds for the GridVars
    hsize_t n1 = pmb->cellbounds.ncellsi(domain);
    hsize_t n2 = pmb->cellbounds.ncellsj(domain);
    hsize_t n3 = pmb->cellbounds.ncellsk(domain);
    // TODO starting location in the global?  Only accessible/calculable without refinement

    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    hdf5_open(fname.c_str());

    // Read everything from root
    hdf5_set_directory("/");

    hid_t string_type = hdf5_make_str_type(20);
    char version[20];
    hdf5_read_single_val(version, "version", string_type);
    if (MPIRank0())
    {
        cout << "Restarting from " << fname << ", file version " << version << endl << endl;
    }

    // First check size
    int n1file, n2file, n3file;
    hdf5_read_single_val(&n1file, "n1", H5T_STD_I32LE);
    hdf5_read_single_val(&n2file, "n2", H5T_STD_I32LE);
    hdf5_read_single_val(&n3file, "n3", H5T_STD_I32LE);
    if (n1file != n1tot || n2file != n2tot || n3file != n3tot)
    {
        throw invalid_argument(string_format("Restart file is wrong size: %d %d %d!\nMesh size: %d %d %d",
                    n1file, n2file, n3file, n1tot, n2tot, n3tot));
    }

    double gam;
    hdf5_read_single_val(&gam, "gam", H5T_IEEE_F64LE);
    double par_gam = pmb->packages["GRMHD"]->Param<Real>("gamma");
    if (abs(gam - par_gam) > 0.01) {
        throw invalid_argument(string_format("Expected gamma of %f but parameters specify %f", gam, par_gam));
    }
    // It's *incredibly* weird parthenon doesn't allow this yet
    //if(MPIRank0()) cout << "Setting fluid gamma from restart: " << gam << endl;
    //pmb->packages["GRMHD"]->AllParams().SetParam<Real>("gamma", gam);

    Real tf;
    hdf5_read_single_val(&tf, "tf", H5T_IEEE_F64LE);

    // TODO check this, and eventually create the grid/mesh after this...
    // if (hdf5_exists("Rin")) {
    //     hdf5_read_single_val(&Rin, "Rin", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&Rout, "Rout", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&hslope, "hslope", H5T_IEEE_F64LE);
    // } else {
    //     hdf5_read_single_val(&x1Min, "x1Min", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&x1Max, "x1Max", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&x2Min, "x2Min", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&x2Max, "x2Max", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&x3Min, "x3Min", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&x3Max, "x3Max", H5T_IEEE_F64LE);
    // }
    // TODO are any of these really needed?  The codes are too different and no old runs have useful e-
    // hdf5_read_single_val(&t, "t", H5T_IEEE_F64LE);
    // hdf5_read_single_val(&nstep, "nstep", H5T_STD_I32LE);
    // if(hdf5_exists("game")) {
    //     hdf5_read_single_val(&game, "game", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&gamp, "gamp", H5T_IEEE_F64LE);
    //     hdf5_read_single_val(&fel0, "fel0", H5T_IEEE_F64LE);
    // }
    // hdf5_read_single_val(&restart_id, "restart_id", H5T_STD_I32LE);
    // hdf5_read_single_val(&dump_cnt, "dump_cnt", H5T_STD_I32LE);
    // hdf5_read_single_val(&dt, "dt", H5T_IEEE_F64LE);

    hsize_t nfprim;
    if(hdf5_exists("game")) {
        nfprim = NPRIM + 2;
    } else {
        nfprim = NPRIM;
    }

    // Declare known sizes for outputting primitives
    static hsize_t fdims[] = {nfprim, n3tot, n2tot, n1tot};
    static hsize_t fcount[] = {nfprim, n3, n2, n1};

    // TODO figure out single restart -> multi mesh
    //hsize_t fstart[] = {0, global_start[2], global_start[1], global_start[0]};
    hsize_t fstart[] = {0, 0, 0, 0};

    // These are dimensions for memory,
    static hsize_t mdims[] = {nfprim, n3, n2, n1};
    static hsize_t mstart[] = {0, 0, 0, 0};

    Real *ptmp = new Real[nfprim*n3*n2*n1];
    hdf5_read_array(ptmp, "p", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);

    // End HDF5 reads
    hdf5_close();

    auto Phost = P.GetHostMirror();

    // Host-side copy into the mirror
    Kokkos::parallel_for("copy_restart_state", Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<4>>({0, ks, js, is}, {NPRIM, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA_VARS {
            Phost(p, k, j, i) = ptmp[p*n3*n2*n1 + (k-NGHOST)*n2*n1 + (j-NGHOST)*n1 + (i-NGHOST)];
        }
    );
    delete[] ptmp;

    // Deep copy to device
    P.DeepCopy(Phost);

    // Every iharm3d sim we'd be restarting had these
    outflow_x1(G, P, n1, n2, n3);
    polar_x2(G, P, n1, n2, n3);
    periodic_x3(G, P, n1, n2, n3);

    return tf;
}

// Boundary functions for the initial sync
// Many possible speed improvements but these are run once & shouldn't exist at all
void outflow_x1(const GRCoordinates& G, GridVars P, int n1, int n2, int n3)
{
    Kokkos::parallel_for("outflow_x1_l", MDRangePolicy<Rank<3>>({NGHOST, NGHOST, 0}, {n3+NGHOST, n2+NGHOST, NGHOST}),
        KOKKOS_LAMBDA_3D {
            int iz = NGHOST;

            PLOOP P(p, k, j, i) = P(p, k, j, iz);

            double rescale = G.gdet(Loci::center, j, iz) / G.gdet(Loci::center, j, i);
            P(prims::B1, k, j, i) *= rescale;
            P(prims::B2, k, j, i) *= rescale;
            P(prims::B3, k, j, i) *= rescale;
        }
    );
    Kokkos::parallel_for("outflow_x1_r", MDRangePolicy<Rank<3>>({NGHOST, NGHOST, n1+NGHOST}, {n3+NGHOST, n2+NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
            int iz = n1 + NGHOST - 1;

            PLOOP P(p, k, j, i) = P(p, k, j, iz);

            double rescale = G.gdet(Loci::center, j, iz) / G.gdet(Loci::center, j, i);
            P(prims::B1, k, j, i) *= rescale;
            P(prims::B2, k, j, i) *= rescale;
            P(prims::B3, k, j, i) *= rescale;
        }
    );
}

void polar_x2(const GRCoordinates& G, GridVars P, int n1, int n2, int n3)
{
    Kokkos::parallel_for("reflect_x2_l", MDRangePolicy<Rank<3>>({NGHOST, 0, 0}, {n3+NGHOST, NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
          // Reflect across NG.  The zone j is (NG-j) prior to reflection,
          // set it equal to the zone that far *beyond* NG
          int jrefl = NGHOST + (NGHOST - j) - 1;
          PLOOP P(p, k, j, i) = P(p, k, jrefl, i);

          // TODO These are suspect...
          P(prims::u2, k, j, i) *= -1.;
          P(prims::B2, k, j, i) *= -1.;
        }
    );
    Kokkos::parallel_for("reflect_x2_r", MDRangePolicy<Rank<3>>({NGHOST, n2+NGHOST, 0}, {n3+NGHOST, n2+2*NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
          // Reflect across (NG+N2).  The zone j is (j - (NG+N2)) after reflection,
          // set it equal to the zone that far *before* (NG+N2)
          int jrefl = (NGHOST + n2) - (j - (NGHOST + n2)) - 1;
          PLOOP P(p, k, j, i) = P(p, k, jrefl, i);

          // TODO These are suspect...
          P(prims::u2, k, j, i) *= -1.;
          P(prims::B2, k, j, i) *= -1.;
        }
    );
}

void periodic_x3(const GRCoordinates& G, GridVars P, int n1, int n2, int n3)
{
    Kokkos::parallel_for("periodic_x3_l", MDRangePolicy<Rank<3>>({0, 0, 0}, {NGHOST, n2+2*NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
            int kz = k + n3;

            PLOOP P(p, k, j, i) = P(p, kz, j, i);
        }
    );
    Kokkos::parallel_for("periodic_x3_r", MDRangePolicy<Rank<3>>({n3+NGHOST, 0, 0}, {n3+2*NGHOST, n2+2*NGHOST, n1+2*NGHOST}),
        KOKKOS_LAMBDA_3D {
            int kz = k - n3;

            PLOOP P(p, k, j, i) = P(p, kz, j, i);
        }
    );
}