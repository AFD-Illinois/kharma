/*
 * Everything that doesn't fit somewhere else.  General C/C++ convenience functions. 
 */
#pragma once

#include "decs.hpp"

#include <memory>
#include <string>
#include <stdexcept>
#include <chrono>

// Some crazy macros for std::chrono
// TODO make this less scary & weird
#if DEBUG
#define PRINT_SEC(x) std::chrono::duration_cast<std::chrono::duration<double>>(x).count()
#define TIME_NOW std::chrono::high_resolution_clock::now()
#else
#define PRINT_SEC(x)
#define TIME_NOW 0.
#endif

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

using namespace std; // This allows CUDA to override max & min
// Thanks https://stackoverflow.com/questions/9323903/most-efficient-elegant-way-to-clip-a-number
template <typename T>
KOKKOS_INLINE_FUNCTION T clip(const T& n, const T& lower, const T& upper) {
  return max(lower, min(n, upper));
}