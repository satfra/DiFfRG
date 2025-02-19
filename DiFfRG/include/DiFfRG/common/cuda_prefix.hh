#pragma once

#ifdef __CUDACC__

// external libraries
#include "cuda.h"
#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <thrust/memory.h>

namespace DiFfRG
{
  /**
   * @brief Check if a CUDA error occurred and print an error message if it did.
   *
   * @param prefix A prefix to be printed before the error message.
   */
  void check_cuda(std::string prefix = "");
} // namespace DiFfRG

#else

// Forward declare a few classes to make the compiler happy.
namespace thrust
{
  template <typename T> class device_vector;
}
struct cudaArray;
using cudaArray_t = cudaArray *;
using cudaTextureObject_t = unsigned long long;

#ifndef __forceinline__
#define __forceinline__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __constant__
#define __constant__
#endif

#endif