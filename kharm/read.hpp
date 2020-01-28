/*
 * HDF5 functions for K/HARM
 */

#include "decs.hpp"

using namespace std;

#include "H5Cpp.h"
using namespace H5;

#include "highfive/H5Selection.hpp"

template <typename Scalar>
Scalar readScalar(H5File file, string dset_name)
{
    // Get the dataset object
    DataSet dataset = file.openDataSet(dset_name);
    //return dataset.read();
    return 0.0;
}

Kokkos::View<Real****, Kokkos::HostSpace> readDataset4(H5File file, string dset_name, int ng)
{
    // Get the dataset object
    DataSet dataset = file.openDataSet(dset_name);

    DataSpace dataspace = dataset.getSpace();

    hsize_t dims[4];
    dataspace.getSimpleExtentDims(dims, NULL);

    for (int dim=0; dim < 4; ++dim)
    {
        cout << (unsigned long)(dims[dim]);
        if (dim < 4 - 1)
            cout << " x ";
    }
    cout << endl;

    // TODO load dims vs fdims, offset based on MPI
    hsize_t offset[4] = {0};
    dataspace.selectHyperslab(H5S_SELECT_SET, dims, offset);

    hsize_t dimsm[4];
    for (int dim=0; dim < 4; ++dim)
    {
        dimsm[dim] = dims[dim] + 2 * ng;
    }
    DataSpace memspace(4, dimsm);

    hsize_t offsetm[4] = {ng};
    memspace.selectHyperslab(H5S_SELECT_SET, dims, offsetm);


    Kokkos::View<Real****, Kokkos::HostSpace> data_out(dset_name, dims[0], dims[1], dims[2], dims[3]);
    //dataset.read(data_out.data(), PredType::NATIVE_DOUBLE, memspace, dataspace);
    return data_out;
}