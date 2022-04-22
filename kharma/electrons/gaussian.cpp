#include "problem.hpp"
// #include "/usr/lib/hpc/gnu7/mpich/fftw3/3.3.9/include/fftw3-mpi.h"
#include "fftw3-mpi.h"
#include <cmath>
#include <random>
using namespace std;

float normalRand() {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0,1};
    return d(gen);
}


void create_grf(int Nx1, int Nx2, double lx1, double lx2, 
                    double * dv1, double * dv2) {
    double dkx1 = 2*M_PI/lx1;
    double dkx2 = 2*M_PI/lx2;
    double Dx1 = lx1/Nx1;
    double Dx2 = lx2/Nx2;

    double kx1max = 2*M_PI/(2*Dx1);
    double kx2max = 2*M_PI/(2*Dx2);
    double k_peak = 4*M_PI/lx1;

    fftw_complex *dvkx1, *dvkx2;
    dvkx1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx1 * Nx2);
    dvkx2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx1 * Nx2);
#pragma omp parallel for simd collapse(2)
    for (size_t i = 0; i < Nx1 ; i ++) {
        for (size_t j = 0; j < Nx2 ; j ++) {
            double retx1 = i * dkx1;
            double retx2 = j * dkx2;
            if(retx1 > kx1max) retx1 = retx1 - 2*kx1max;
            if(retx2 > kx2max) retx2 = retx2 - 2*kx2max;
            double curr_k_magn = pow(pow(retx1, 2) + pow(retx2, 2), 0.5);

            double pwr_spct = pow(curr_k_magn, 6)*exp(-8*curr_k_magn/k_peak);
            if (curr_k_magn != 0) {
                retx1 /= curr_k_magn;
                retx2 /= curr_k_magn;
            }

            double noisy_dvkx1_real = pwr_spct*normalRand(); double noisy_dvkx1_imag = pwr_spct*normalRand();
            double noisy_dvkx2_real = pwr_spct*normalRand(); double noisy_dvkx2_imag = pwr_spct*normalRand();

            //real part of kx, using real part of dot product, and the kx component. second line is imag part
            double adj_dvkx1_real = (retx1*noisy_dvkx1_real + retx2*noisy_dvkx2_real)*retx1;
            double adj_dvkx1_imag = (retx1*noisy_dvkx1_imag + retx2*noisy_dvkx2_imag)*retx1;
            double adj_dvkx2_real = (retx1*noisy_dvkx1_real + retx2*noisy_dvkx2_real)*retx2;
            double adj_dvkx2_imag = (retx1*noisy_dvkx1_imag + retx2*noisy_dvkx2_imag)*retx2;

            dvkx1[i*Nx1+j][0] = noisy_dvkx1_real - adj_dvkx1_real;  dvkx1[i*Nx1+j][1] = noisy_dvkx1_imag - adj_dvkx1_imag;
            dvkx2[i*Nx1+j][0] = noisy_dvkx2_real - adj_dvkx2_real;  dvkx2[i*Nx1+j][1] = noisy_dvkx2_imag - adj_dvkx2_imag;
        }
    }

    fftw_plan p_x1, p_x2;
    p_x1 = fftw_plan_dft_2d(Nx1, Nx2, dvkx1, dvkx1, FFTW_BACKWARD, FFTW_ESTIMATE); //in-place
    p_x2 = fftw_plan_dft_2d(Nx1, Nx2, dvkx2, dvkx2, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_x1); //look for threads documentation
    fftw_execute(p_x2);

    fftw_destroy_plan(p_x1);  fftw_destroy_plan(p_x2);
    double total_dv1 = 0.0; double total_dv2 = 0.0; //work for doubles?
#pragma omp parallel for simd collapse(2) reduction(+:total_dv1, total_dv2)
    for (size_t i = 0; i < Nx1 ; i ++) {
        for (size_t j = 0; j < Nx2 ; j ++) {
            total_dv1 += dvkx1[i*Nx1+j][0];
            total_dv2 += dvkx2[i*Nx1+j][0];
        }
    }
    total_dv1 /= Nx1; total_dv2 /= Nx2;
#pragma omp parallel for simd collapse(2)
    for (size_t i = 0; i < Nx1 ; i ++) {
        for (size_t j = 0; j < Nx2 ; j ++) {
            dv1[i*Nx1+j] = dvkx1[i*Nx1+j][0] - total_dv1;
            dv2[i*Nx1+j] = dvkx2[i*Nx1+j][0] - total_dv2;
        }
    } //centered!
    fftw_free(dvkx1);   fftw_free(dvkx2);
    // print(dv1, Nx1, Nx2); print(dv2, Nx1, Nx2); //bad option to compare grf among iterations
}
