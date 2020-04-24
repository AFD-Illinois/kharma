
#include "decs.hpp"

using namespace parthenon;

void count_print_flags(MeshBlock *pmb, const ParArrayND<int> pflag)
{
    int n_tot = 0, n_neg_in = 0, n_max_iter = 0;
    int n_utsq = 0, n_gamma = 0, n_neg_u = 0, n_neg_rho = 0, n_neg_both = 0;

    for(int i=0; i<pmb->ncells1; ++i)
        for(int j=0; j<pmb->ncells2; ++j)
            for(int k=0; k<pmb->ncells3; ++k)
    {
            if (pflag(i, j, k) != InversionStatus::success) ++n_tot;
            if (pflag(i, j, k) == InversionStatus::neg_input) ++n_neg_in;
            if (pflag(i, j, k) == InversionStatus::max_iter) ++n_max_iter;
            if (pflag(i, j, k) == InversionStatus::bad_ut) ++n_utsq;
            if (pflag(i, j, k) == InversionStatus::bad_gamma) ++n_gamma;
            if (pflag(i, j, k) == InversionStatus::neg_rho) ++n_neg_rho;
            if (pflag(i, j, k) == InversionStatus::neg_u) ++n_neg_u;
            if (pflag(i, j, k) == InversionStatus::neg_rhou) ++n_neg_both;
    }

    cerr << "PFLAGS:" << n_tot << endl;
    cerr << "Negative input: " << n_neg_in << endl;
    cerr << "Hit max iter: " << n_max_iter << endl;
    cerr << "Velocity invalid: " << n_utsq << endl;
    cerr << "Gamma invalid: " << n_gamma << endl;
    cerr << "Negative rho: " << n_neg_rho << endl;
    cerr << "Negative U: " << n_neg_u << endl;
    cerr << "Negative rho & U: " << n_neg_both << endl << endl;
}