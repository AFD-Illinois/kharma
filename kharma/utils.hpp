/*
 * Everything that doesn't fit somewhere else.  General C/C++ convenience functions. 
 */
#pragma once

#include "decs.hpp"

#include <memory>
#include <string>
#include <stdexcept>

// Thanks https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] ); 
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

// The world needed these
// Maybe not in this form
KOKKOS_INLINE_FUNCTION void print_matrix(std::string name, double g[GR_DIM][GR_DIM], bool kill_on_nan=false)
{
    // Print a name and a matrix
    printf("%s:\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n", name.c_str(),
            g[0][0], g[0][1], g[0][2], g[0][3], g[1][0], g[1][1], g[1][2],
            g[1][3], g[2][0], g[2][1], g[2][2], g[2][3], g[3][0], g[3][1],
            g[3][2], g[3][3]);

    if (kill_on_nan) {
        // Additionally kill things if/when we hit NaNs
        DLOOP2 if (isnan(g[mu][nu])) exit(-1);
    }
}
KOKKOS_INLINE_FUNCTION void print_vector(std::string name, double v[GR_DIM], bool kill_on_nan=false)
{
    printf("%s: %g\t%g\t%g\t%g\n", name.c_str(), v[0], v[1], v[2], v[3]);

    if (kill_on_nan) {
        DLOOP2 if (isnan(v[nu])) exit(-1);
    }
}

using namespace std; // This allows CUDA to override max & min
// Thanks https://stackoverflow.com/questions/9323903/most-efficient-elegant-way-to-clip-a-number
template <typename T>
KOKKOS_INLINE_FUNCTION T clip(const T& n, const T& lower, const T& upper) {
  return max(lower, min(n, upper));
}