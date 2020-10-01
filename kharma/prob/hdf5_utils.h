/* 
 *  File: hdf5_utils.h
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *  
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <hdf5.h>

// Define a debug flag to print each read/write
//#define DEBUG 1

// Force MPI on or off
//#define USE_MPI 0

// Blob "copy" utility
typedef hid_t hdf5_blob;
hdf5_blob hdf5_get_blob(const char *name);
int hdf5_write_blob(hdf5_blob blob, const char *name);
int hdf5_close_blob(hdf5_blob blob);

// File
int hdf5_create(const char *fname);
int hdf5_open(const char *fname);
int hdf5_close();

// Directory
int hdf5_make_directory(const char *name);
void hdf5_set_directory(const char *path);

// Write
int hdf5_write_single_val(const void *val, const char *name, hsize_t hdf5_type);
int hdf5_write_array(const void *data, const char *name, size_t rank,
                      hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type);

// Read
int hdf5_exists(const char *name);
int hdf5_read_single_val(void *val, const char *name, hsize_t hdf5_type);
int hdf5_read_array(void *data, const char *name, size_t rank,
                      hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type);

// Convenience and annotations
hid_t hdf5_make_str_type(size_t len);
int hdf5_write_str_list(const void *data, const char *name, size_t strlen, size_t len);
int hdf5_add_attr(const void *att, const char *att_name, const char *data_name, hsize_t hdf5_type);
int hdf5_add_units(const char *name, const char *unit);

