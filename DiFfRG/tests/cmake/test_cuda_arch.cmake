# Unit tests for the DiFfRG_CUDA_ARCH helpers (cmake/cuda_arch.cmake).
#
# Run with `cmake -P`, so they need neither a GPU nor a configured build tree:
#
#   cmake -DCASE=flags -P test_cuda_arch.cmake
#
# CASE selects a group; "invalid" is expected to abort, and is registered with
# WILL_FAIL so that a helper which silently accepted nonsense would be caught.

cmake_minimum_required(VERSION 3.20)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/cuda_arch.cmake")

function(expect what expected actual)
  if(NOT "${expected}" STREQUAL "${actual}")
    message(FATAL_ERROR "${what}: expected [${expected}], got [${actual}]")
  endif()
endfunction()

# Normalize a list, then turn it into flags -- the two steps always run together.
function(arch_flags out)
  _diffrg_normalize_cuda_arch(_ccs ${ARGN})
  _diffrg_cuda_arch_flags(_flags ${_ccs})
  set(${out} "${_flags}" PARENT_SCOPE)
endfunction()

if(CASE STREQUAL "normalize")
  _diffrg_normalize_cuda_arch(_r 90)
  expect("plain" "90" "${_r}")
  _diffrg_normalize_cuda_arch(_r 9.0)
  expect("dotted" "90" "${_r}")
  _diffrg_normalize_cuda_arch(_r " 8.9 ")
  expect("padded" "89" "${_r}")
  _diffrg_normalize_cuda_arch(_r 12.0)
  expect("three digits" "120" "${_r}")
  _diffrg_normalize_cuda_arch(_r 9.0 8.0 9.0 7.5)
  expect("sorted and deduplicated" "75;80;90" "${_r}")
  # NATURAL rather than lexicographic ordering: 120 must not sort below 75.
  _diffrg_normalize_cuda_arch(_r 12.0 7.5)
  expect("numeric order" "75;120" "${_r}")
  _diffrg_normalize_cuda_arch(_r "")
  expect("empty" "" "${_r}")

elseif(CASE STREQUAL "flags")
  arch_flags(_r 9.0)
  expect("single" "-arch=sm_90" "${_r}")
  # Several architectures go through one -arch plus one -code: a second -arch or
  # any -gencode makes the bundle's nvcc_wrapper abort.
  arch_flags(_r 9.0 8.0)
  expect("multiple" "-arch=compute_80;-code=sm_80,sm_90,compute_80" "${_r}")
  arch_flags(_r 9.0 8.0 7.5)
  expect("three" "-arch=compute_75;-code=sm_75,sm_80,sm_90,compute_75" "${_r}")
  arch_flags(_r "")
  expect("empty" "" "${_r}")

  # Whatever we emit, exactly one -arch and never a -gencode.
  foreach(_case "90" "90;80" "75;80;90;120")
    arch_flags(_flags ${_case})
    string(REGEX MATCHALL "-arch=" _arches "${_flags}")
    list(LENGTH _arches _n)
    expect("one -arch for [${_case}]" "1" "${_n}")
    if(_flags MATCHES "-gencode")
      message(FATAL_ERROR "[${_case}] emitted a -gencode: ${_flags}")
    endif()
  endforeach()

elseif(CASE STREQUAL "replace")
  arch_flags(_flags 9.0)
  _diffrg_replace_cuda_arch(_r "${_flags}" "-extended-lambda;-arch=sm_75")
  expect("plain list" "-extended-lambda;-arch=sm_90" "${_r}")

  # Kokkos exports its options inside a generator expression; the surrounding
  # genex has to survive the rewrite.
  _diffrg_replace_cuda_arch(
    _r "${_flags}" "$<$<COMPILE_LANGUAGE:CXX>:-extended-lambda;-arch=sm_75>")
  expect("inside a genex" "$<$<COMPILE_LANGUAGE:CXX>:-extended-lambda;-arch=sm_90>" "${_r}")

  arch_flags(_flags 9.0 8.0)
  _diffrg_replace_cuda_arch(_r "${_flags}" "-arch=sm_75")
  expect("multi-arch" "-arch=compute_80;-code=sm_80,sm_90,compute_80" "${_r}")

  # A property with no architecture flag (a CPU-only bundle) reports "no change"
  # rather than being rewritten into something.
  _diffrg_replace_cuda_arch(_r "${_flags}" "-extended-lambda;-O2")
  expect("nothing to replace" "" "${_r}")

elseif(CASE STREQUAL "invalid")
  # Must abort: registered with WILL_FAIL.
  _diffrg_normalize_cuda_arch(_r "${BAD}")
  message(FATAL_ERROR "accepted '${BAD}' as a compute capability")

else()
  message(FATAL_ERROR "unknown CASE '${CASE}'")
endif()
