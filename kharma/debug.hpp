// Debug tools
#pragma once

#include "decs.hpp"

#include "mesh/mesh.hpp"

using namespace std;

template<typename T>
void CountPFlags(MeshBlock *pmb, T pflag, IndexDomain domain=IndexDomain::entire)
{
    int n_tot = 0, n_neg_in = 0, n_max_iter = 0;
    int n_utsq = 0, n_gamma = 0, n_neg_u = 0, n_neg_rho = 0, n_neg_both = 0;

    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // TODO par_for
    for(int k=ks; k <= ke; ++k)
        for(int j=js; j <= je; ++j)
            for(int i=is; i <= ie; ++i)
    {
        int flag = pflag(k, j, i);
        if (flag != InversionStatus::success) ++n_tot;
        if (flag == InversionStatus::neg_input) ++n_neg_in;
        if (flag == InversionStatus::max_iter) ++n_max_iter;
        if (flag == InversionStatus::bad_ut) ++n_utsq;
        if (flag == InversionStatus::bad_gamma) ++n_gamma;
        if (flag == InversionStatus::neg_rho) ++n_neg_rho;
        if (flag == InversionStatus::neg_u) ++n_neg_u;
        if (flag == InversionStatus::neg_rhou) ++n_neg_both;
    }

    // TODO MPI

    cerr << "PFLAGS: " << n_tot << endl;
    if (n_neg_in > 0) cerr << "Negative input: " << n_neg_in << endl;
    if (n_max_iter > 0) cerr << "Hit max iter: " << n_max_iter << endl;
    if (n_utsq > 0) cerr << "Velocity invalid: " << n_utsq << endl;
    if (n_gamma > 0) cerr << "Gamma invalid: " << n_gamma << endl;
    if (n_neg_rho > 0) cerr << "Negative rho: " << n_neg_rho << endl;
    if (n_neg_u > 0) cerr << "Negative U: " << n_neg_u << endl;
    if (n_neg_both > 0) cerr << "Negative rho & U: " << n_neg_both << endl;
    cerr << endl;
}

template<typename T>
void CountFFlags(MeshBlock *pmb, T fflag, IndexDomain domain=IndexDomain::interior)
{
    int n_tot = 0, n_geom_rho = 0, n_geom_u = 0, n_b_rho = 0, n_b_u = 0, n_temp = 0, n_gamma = 0, n_ktot = 0;

    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // TODO par_for
    for(int k=ks; k <= ke; ++k)
        for(int j=js; j <= je; ++j)
            for(int i=is; i <= ie; ++i)
    {
        int flag = fflag(k, j, i);
        if (flag != 0) n_tot++;
        if (flag & HIT_FLOOR_GEOM_RHO) n_geom_rho++;
        if (flag & HIT_FLOOR_GEOM_U) n_geom_u++;
        if (flag & HIT_FLOOR_B_RHO) n_b_rho++;
        if (flag & HIT_FLOOR_B_U) n_b_u++;
        if (flag & HIT_FLOOR_TEMP) n_temp++;
        if (flag & HIT_FLOOR_GAMMA) n_gamma++;
        if (flag & HIT_FLOOR_KTOT) n_ktot++;
    }

    // n_geom_rho = mpi_reduce_int(n_geom_rho);
    // n_geom_u = mpi_reduce_int(n_geom_u);
    // n_b_rho = mpi_reduce_int(n_b_rho);
    // n_b_u = mpi_reduce_int(n_b_u);
    // n_temp = mpi_reduce_int(n_temp);
    // n_gamma = mpi_reduce_int(n_gamma);
    // n_ktot = mpi_reduce_int(n_ktot);

    cerr << "FLOORS: " << n_tot << endl;
    if (n_geom_rho > 0) cerr << "GEOM_RHO: " << n_geom_rho << endl;
    if (n_geom_u > 0) cerr << "GEOM_U: " << n_geom_u << endl;
    if (n_b_rho > 0) cerr << "B_RHO: " << n_b_rho << endl;
    if (n_b_u > 0) cerr << "B_U: " << n_b_u << endl;
    if (n_temp > 0) cerr << "TEMPERATURE: " << n_temp << endl;
    if (n_gamma > 0) cerr << "GAMMA: " << n_gamma << endl;
    if (n_ktot > 0) cerr << "KTOT: " << n_ktot << endl;
    cerr << endl;
}

double MaxDivB(Container<Real>& rc);