#pragma once

// Force-included into every C++ translation unit of DiFfRG and of every consumer
// of the exported target (see setup_target() in cmake/setup_build_system.cmake).
// It collects the workarounds that have to be in effect before any other header
// is parsed, so it has to stay self-contained and cheap.
//
// Only a single -include option may be used: CMake de-duplicates the repeated
// "-include" token of a second one and orphans its file name, which the compiler
// then treats as an additional input file.

// deal.II's tensor.h uses assert() without including <cassert>.
#include <cassert>

// nvcc, observed with CUDA 13.x driving the libstdc++ 13 headers in C++20 mode,
// mis-handles the constrained partial specialization
//
//   template <typename T> requires std::is_object_v<T> struct iterator_traits<T *>;
//
// Its frontend fails to select the specialization when std::iterator_traits<char *>
// is first required from a deferred parsing context -- in particular from a default
// argument that materialises a std::string, which is exactly what deal.II's
// DeclExceptionMsg macro expands to:
//
//   Exception(const std::string &msg = "some text") : arg(msg) {}
//
// Every translation unit that pulls in <deal.II/base/exceptions.h> before the
// specialization has been completed then drowns in "incomplete type
// std::iterator_traits<char *>" errors from <bits/stl_iterator.h>. Completing the
// specialization eagerly, ahead of every other header, makes the frontend record
// the correct instantiation and the errors disappear.
#if defined(__CUDACC__)

#include <iterator>

namespace DiFfRG
{
  namespace detail
  {
    using nvcc_iterator_traits_fix = std::iterator_traits<char *>::value_type;
  }
} // namespace DiFfRG

#endif
