# ##############################################################################
# CUDA architecture helpers
# ##############################################################################
#
# Pure helpers behind DiFfRG_CUDA_ARCH; see setup_build_system.cmake for what
# they are used for. They live in their own file so the tests can exercise them
# with `cmake -P`, without a bundle or a compiler.

# Normalize a user- or probe-supplied capability list into sorted, deduplicated
# two-or-more-digit form: "8.9;9.0" and "89;90" both become "89;90".
function(_diffrg_normalize_cuda_arch out)
  set(_ccs "")
  foreach(_cc IN LISTS ARGN)
    string(STRIP "${_cc}" _cc)
    string(REPLACE "." "" _cc "${_cc}")
    if(_cc STREQUAL "")
      continue()
    endif()
    if(NOT _cc MATCHES "^[0-9][0-9]+$")
      message(
        FATAL_ERROR
          "DiFfRG_CUDA_ARCH: '${_cc}' is not a CUDA compute capability. Use the "
          "two-digit form (80, 89, 90) or the dotted form (8.0, 8.9, 9.0).")
    endif()
    list(APPEND _ccs "${_cc}")
  endforeach()
  list(REMOVE_DUPLICATES _ccs)
  list(SORT _ccs COMPARE NATURAL)
  set(${out} "${_ccs}" PARENT_SCOPE)
endfunction()

# Ask the driver which GPUs are present. Fails (leaving the output empty) on
# driverless build nodes, which is why the value can also be given by hand.
function(_diffrg_detect_cuda_arch out)
  set(${out} "" PARENT_SCOPE)
  find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
  if(NOT NVIDIA_SMI_EXECUTABLE)
    return()
  endif()
  execute_process(
    COMMAND "${NVIDIA_SMI_EXECUTABLE}" --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _out
    ERROR_QUIET
    RESULT_VARIABLE _rc
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _rc EQUAL 0)
    return()
  endif()
  string(REPLACE "\n" ";" _out "${_out}")
  _diffrg_normalize_cuda_arch(_ccs ${_out})
  set(${out} "${_ccs}" PARENT_SCOPE)
endfunction()

# Turn a capability list into nvcc flags.
#
# Exactly one "-arch=" token is produced, and never a "-gencode": applications
# compile through the bundle's nvcc_wrapper, which aborts ("ARCH is being set
# twice with different flags") as soon as it sees a second -arch or -gencode.
# Its -code handling carries no such guard, so several real architectures are
# requested through a single -code list instead. Both forms give native SASS for
# every listed architecture plus a PTX fallback for future ones.
function(_diffrg_cuda_arch_flags out)
  list(LENGTH ARGN _n)
  if(_n EQUAL 0)
    set(${out} "" PARENT_SCOPE)
    return()
  endif()
  list(GET ARGN 0 _min)
  if(_n EQUAL 1)
    set(${out} "-arch=sm_${_min}" PARENT_SCOPE)
    return()
  endif()
  set(_codes "")
  foreach(_cc IN LISTS ARGN)
    list(APPEND _codes "sm_${_cc}")
  endforeach()
  list(APPEND _codes "compute_${_min}")
  list(JOIN _codes "," _codes)
  set(${out} "-arch=compute_${_min};-code=${_codes}" PARENT_SCOPE)
endfunction()

# Swap the bundle's "-arch=sm_XX" for ${flags} in an option list, leaving a list
# that carries no such flag untouched. Done textually so that an option wrapped
# in a generator expression -- which is how Kokkos exports its own -- keeps its
# surrounding genex intact.
function(_diffrg_replace_cuda_arch out flags opts)
  if(NOT opts MATCHES "-arch=sm_[0-9]+")
    set(${out} "" PARENT_SCOPE)
    return()
  endif()
  string(REGEX REPLACE "-arch=sm_[0-9]+" "${flags}" opts "${opts}")
  set(${out} "${opts}" PARENT_SCOPE)
endfunction()

# Retarget one interface property of an imported target. Rewriting rather than
# appending is what keeps nvcc_wrapper happy, and the link line carries the same
# flag, so it has to stay in step with the compile line.
function(_diffrg_retarget_cuda_arch target property flags)
  if(NOT TARGET ${target})
    return()
  endif()
  get_target_property(_opts ${target} ${property})
  if(NOT _opts)
    return()
  endif()
  _diffrg_replace_cuda_arch(_opts "${flags}" "${_opts}")
  if(_opts)
    set_target_properties(${target} PROPERTIES ${property} "${_opts}")
  endif()
endfunction()