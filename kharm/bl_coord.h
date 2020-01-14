/*
 * Utility functions for Boyer-Lindquist coordinates
 * Provided for problem setups but not otherwise used in core functions
 */

#pragma once

#include "decs.h"

struct of_geom {
    double gcon[NDIM][NDIM];
    double gcov[NDIM][NDIM];
    double g;
    double alpha;
};

// Boyer-Lindquist metric functions

void blgset(int i, int j, struct of_geom *geom);

void bl_gcov_func(double r, double th, double gcov[NDIM][NDIM]);
void bl_gcon_func(double r, double th, double gcon[NDIM][NDIM]);
double bl_gdet_func(double r, double th);

void bl_to_ks(double X[NDIM], double ucon_bl[NDIM], double ucon_ks[NDIM]);

void coord_transform(struct GridGeom *G, struct FluidState *S, int i, int j, int k);

