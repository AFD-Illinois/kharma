/*
 *
 */
#pragma once

#include <string>

#include <hdf5.h>

// Crash on read/write failures.  Saves checking return values like a pleb
#ifndef FAIL_HARD
#define FAIL_HARD 1
#endif

#if FAIL_HARD
#define FAIL(errcode, fn_name, val_name) { fprintf(stderr, "HDF5 Error code %d in %s processing %s\n", (int) errcode, fn_name, val_name); exit(-1); }
#else
#define FAIL(errcode, fn_name, val_name) return errcode;
#endif

#define STRLEN 2048

// TODO templates, exceptions.  Make C++
// Probably also split some things out: make_str_type
// Finally, bring back blobs?
class H5F {
    public:
        H5F(){};
        int create(const char *fname);
        int open(const char *fname);
        int close();
        int make_directory(const char *name);
        void set_directory(const char *path);
        int exists(const char *name);
        hid_t make_str_type(size_t len);
        int add_attr(const void *att, const char *att_name, const char *data_name, hsize_t hdf5_type);
        int add_units(const char *name, const char *unit);
        int write_str_list(const void *data, const char *name, size_t str_len, size_t len);
        int write_array(const void *data, const char *name, size_t rank,
                      hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type);
        int write_single_val(const void *val, const char *name, hsize_t hdf5_type);
        int read_single_val(void *val, const char *name, hsize_t hdf5_type);
        int read_array(void *data, const char *name, size_t rank,
                      hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type);

    protected:
        char cur_dir[STRLEN];
        hid_t file_id;
};

// Create a new HDF file (or overwrite whatever file exists)
int H5F::create(const char *fname)
{
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
#if USE_MPI
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL); // TODO tune HDF with an MPI info object
#endif
  file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  // Everyone expects directory to be root after opening a file
  set_directory("/");

  // Quiet HDF5's own errors, so we can control them
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  if(file_id < 0) FAIL(file_id, "hdf5_create", fname);
  return 0;
}

// Open an existing file for reading
int H5F::open(const char *fname)
{
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
#if USE_MPI
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
#endif
  file_id = H5Fopen(fname, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);

  // Everyone expects directory to be root after open
  set_directory("/");

  // Quiet HDF5's own errors, so we can control them
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  if(file_id < 0) FAIL(file_id, "hdf5_open", fname);
  return 0;
}

// Close a file
int H5F::close()
{
  H5Fflush(file_id,H5F_SCOPE_GLOBAL);
  herr_t err = H5Fclose(file_id);

  if(err < 0) FAIL(err, "hdf5_close", "none");
  return 0;
}

// Make a directory (in the current directory) with given name
// This doesn't take a full path, just a name
int H5F::make_directory(const char *name)
{
  // Add current directory to group name
  char path[STRLEN];
  strncpy(path, cur_dir, STRLEN);
  strncat(path, name, STRLEN - strlen(path));

  if(DEBUG) fprintf(stderr,"Adding dir %s\n", path);

  hid_t group_id = H5Gcreate2(file_id, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (group_id < 0) FAIL(group_id, "hdf5_make_directory", path);
  H5Gclose(group_id);

  return 0;
}

// Set the current directory
void H5F::set_directory(const char *path)
{
  strncpy(cur_dir, path, STRLEN);
}

// Check if an object's name exists: returns 1 if it does
// Doesn't verify if the object itself exists, or anything about it!
int H5F::exists(const char *name) {
  char path[STRLEN];
  strncpy(path, cur_dir, STRLEN);
  strncat(path, name, STRLEN - strlen(path));

  hid_t link_plist = H5Pcreate(H5P_LINK_ACCESS);
  herr_t exists = H5Lexists(file_id, path, link_plist);
  H5Pclose(link_plist);

  if(DEBUG) fprintf(stderr,"Checking existence of %s: %d\n", path, exists);

  return exists > 0;
}

// Return a fixed-size string type
// H5T_VARIABLE indicates any string but isn't compatible w/parallel IO
hid_t H5F::make_str_type(size_t len)
{
  hid_t string_type = H5Tcopy(H5T_C_S1);
  H5Tset_size(string_type, len);
  return string_type;
}

// Add the named attribute to the named variable
int H5F::add_attr(const void *att, const char *att_name, const char *data_name, hsize_t hdf5_type)
{
  char path[STRLEN];
  strncpy(path, cur_dir, STRLEN);
  strncat(path, data_name, STRLEN - strlen(path));

  if(DEBUG) fprintf(stderr,"Adding att %s\n", path);

  hid_t attribute_id = H5Acreate_by_name(file_id, path, att_name, hdf5_type, H5Screate(H5S_SCALAR), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (attribute_id < 0) FAIL(attribute_id, "hdf5_add_attr", path);
  H5Awrite(attribute_id, hdf5_type, att);
  H5Aclose(attribute_id);

  return 0;
}

// Add an attribute named "units"
int H5F::add_units(const char *name, const char *unit)
{
  hid_t string_type = H5Tcopy(H5T_C_S1);
  H5Tset_size(string_type, strlen(unit)+1);
  herr_t err = add_attr(unit, "units", name, string_type);
  if (err < 0) FAIL(err, "hdf5_add_units", name);
  H5Tclose(string_type);
  return 0;
}

// Write a 1D list of strings (used for labeling primitives array)
// Must be an array of constant-length strings, i.e. char strs[len][str_len] = etc.
int H5F::write_str_list(const void *data, const char *name, size_t str_len, size_t len)
{
  char path[STRLEN];
  strncpy(path, cur_dir, STRLEN);
  strncat(path, name, STRLEN - strlen(path));

  if(DEBUG) fprintf(stderr,"Adding str list %s\n", path);

  // Adapted (stolen) from https://support.hdfgroup.org/ftp/HDF5/examples/C/
  hsize_t dims_of_char_dataspace[] = {len};

  hid_t vlstr_h5t = H5Tcopy(H5T_C_S1);
  H5Tset_size(vlstr_h5t, str_len);

  hid_t dataspace = H5Screate_simple(1, dims_of_char_dataspace, NULL);
  hid_t dataset = H5Dcreate(file_id, path, vlstr_h5t, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  herr_t err = H5Dwrite(dataset, vlstr_h5t, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  if (err < 0) FAIL(err, "hdf5_write_str_list", path); // If anything above fails, the write should too

  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Tclose(vlstr_h5t);

  return 0;
}

// Write the section 'mdims_copy' starting at 'mstart' of a C-order array of rank 'rank' and size 'mdims_full'
// To the section 'fdims' at 'fstart' of the file 'file_id'
int H5F::write_array(const void *data, const char *name, size_t rank,
                      hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type)
{
  // Declare spaces of the right size
  hid_t filespace = H5Screate_simple(rank, fdims, NULL);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, fstart, NULL, fcount,
    NULL);
  hid_t memspace = H5Screate_simple(rank, mdims, NULL);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mstart, NULL, fcount,
    NULL);

  // Add our current path to the dataset name
  char path[STRLEN];
  strncpy(path, cur_dir, STRLEN);
  strncat(path, name, STRLEN - strlen(path));

  if(DEBUG) fprintf(stderr,"Adding array %s\n", path);

  // Create the dataset in the file
  hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
  hid_t dset_id = H5Dcreate(file_id, path, hdf5_type, filespace, H5P_DEFAULT,
    plist_id, H5P_DEFAULT);
  H5Pclose(plist_id);

  // Conduct the transfer
  plist_id = H5Pcreate(H5P_DATASET_XFER);
#if USE_MPI
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif
  herr_t err = H5Dwrite(dset_id, hdf5_type, memspace, filespace, plist_id, data);
  if (err < 0) FAIL(err, "hdf5_write_array", path);

  H5Dclose(dset_id);
  H5Pclose(plist_id);
  H5Sclose(filespace);
  H5Sclose(memspace);

  return 0;
}

// Write a single value of hdf5_type to a file
int H5F::write_single_val(const void *val, const char *name, hsize_t hdf5_type)
{
  // Add current path to the dataset name
  char path[STRLEN];
  strncpy(path, cur_dir, STRLEN);
  strncat(path, name, STRLEN - strlen(path));

  if(DEBUG) fprintf(stderr,"Adding val %s\n", path);

  // Declare scalar spaces
  hid_t scalarspace = H5Screate(H5S_SCALAR);
  hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
  hid_t dset_id = H5Dcreate(file_id, path, hdf5_type, scalarspace, H5P_DEFAULT,
    plist_id, H5P_DEFAULT);
  H5Pclose(plist_id);

  // Conduct transfer
  plist_id = H5Pcreate(H5P_DATASET_XFER);
#if USE_MPI
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif
  herr_t err = H5Dwrite(dset_id, hdf5_type, scalarspace, scalarspace, plist_id, val);
  if (err < 0) FAIL(err, "hdf5_write_single_val", path);

  // Close spaces (TODO could definitely keep these open instead of re-declaring)
  H5Dclose(dset_id);
  H5Pclose(plist_id);
  H5Sclose(scalarspace);

  return 0;
}

// These are very like above but there's not a good way to share code...
int H5F::read_single_val(void *val, const char *name, hsize_t hdf5_type)
{
  char path[STRLEN];
  strncpy(path, cur_dir, STRLEN);
  strncat(path, name, STRLEN - strlen(path));

  if(DEBUG) fprintf(stderr,"Reading val %s\n", path);

  hid_t scalarspace = H5Screate(H5S_SCALAR);
  hid_t dset_id = H5Dopen(file_id, path, H5P_DEFAULT);

  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#if USE_MPI
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif
  herr_t err = H5Dread(dset_id, hdf5_type, scalarspace, scalarspace, plist_id, val);
  if (err < 0) FAIL(err, "hdf5_read_single_val", path);

  H5Dclose(dset_id);
  H5Pclose(plist_id);
  H5Sclose(scalarspace);

  return 0;
}

int H5F::read_array(void *data, const char *name, size_t rank,
                      hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type)
{
  hid_t filespace = H5Screate_simple(4, fdims, NULL);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, fstart, NULL, fcount,
    NULL);
  hid_t memspace = H5Screate_simple(4, mdims, NULL);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mstart, NULL, fcount,
    NULL);

  char path[STRLEN];
  strncpy(path, cur_dir, STRLEN);
  strncat(path, name, STRLEN - strlen(path));

  if(DEBUG) fprintf(stderr,"Reading arr %s\n", path);

  hid_t dset_id = H5Dopen(file_id, path, H5P_DEFAULT);

  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#if USE_MPI
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif
  herr_t err = H5Dread(dset_id, hdf5_type, memspace, filespace, plist_id, data);
  if (err < 0) FAIL(err, "hdf5_read_array", path);

  H5Dclose(dset_id);
  H5Pclose(plist_id);
  H5Sclose(filespace);
  H5Sclose(memspace);

  return 0;
}
