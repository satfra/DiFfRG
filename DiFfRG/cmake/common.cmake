if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold "${Esc}[1m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
  set(BoldRed "${Esc}[1;31m")
  set(BoldGreen "${Esc}[1;32m")
  set(BoldYellow "${Esc}[1;33m")
  set(BoldBlue "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan "${Esc}[1;36m")
  set(BoldWhite "${Esc}[1;37m")
endif()

# ##############################################################################
# Target-architecture reporting
# ##############################################################################
#
# Announce which CPU architecture the build targets: "native" (the build
# machine's CPU), "none" (no -march flag, generic baseline), or an explicit
# -march value such as "x86-64-v3". For any real -march, probe the C++
# compiler's predefined macros to list which ISA extensions it enables
# (AVX-512, AVX2, FMA, ...), so it is clear what the resulting binaries will
# require to run -- e.g. an AVX-512 build will SIGILL on a machine without
# AVX-512.
function(diffrg_report_arch _march _cxx)
  if(_march STREQUAL "none" OR _march STREQUAL "")
    message(
      "  ${BoldYellow}[arch] Portable build: no -march flag; targeting a generic CPU.${ColourReset}"
    )
    return()
  endif()

  if(_march STREQUAL "native")
    message(
      "  ${BoldGreen}[arch] Optimizing for the build machine's CPU (-march=native).${ColourReset}"
    )
  else()
    message(
      "  ${BoldGreen}[arch] Targeting -march=${_march}.${ColourReset}")
  endif()

  if(NOT _cxx)
    return()
  endif()
  # Dump the compiler's predefined macros under -march=<value>. Works for
  # GCC/Clang/AppleClang; on anything else (or if the flag is rejected,
  # e.g. some Apple-silicon toolchains) we simply skip the capability list.
  execute_process(
    COMMAND ${_cxx} -march=${_march} -dM -E -x c++ /dev/null
    OUTPUT_VARIABLE _macros
    ERROR_QUIET
    RESULT_VARIABLE _rv)
  if(NOT _rv EQUAL 0)
    message(
      "  ${BoldYellow}    (could not probe -march=${_march} capabilities for this compiler)${ColourReset}"
    )
    return()
  endif()

  # Curated, highest-first list of ISA feature macros worth reporting.
  set(_features
      AVX512F AVX512DQ AVX512BW AVX512VL AVX512CD AVX512VNNI AVX512BF16 AVX2 AVX
      FMA F16C BMI BMI2 POPCNT SSE4_2 SSE4_1 SSSE3 SSE3 SSE2)
  set(_enabled "")
  foreach(_f ${_features})
    if(_macros MATCHES "#define __${_f}__ 1")
      string(REGEX REPLACE "^AVX512" "AVX-512" _name "${_f}")
      string(REGEX REPLACE "^SSE4_" "SSE4." _name "${_name}")
      list(APPEND _enabled "${_name}")
    endif()
  endforeach()

  if(_enabled)
    string(REPLACE ";" ", " _enabled_str "${_enabled}")
    message(
      "  ${BoldGreen}    Enabled CPU features: ${_enabled_str}${ColourReset}")
  endif()
endfunction()

# ##############################################################################
# Target-architecture resolution
# ##############################################################################
#
# Resolve the MARCH/NATIVE pair into a single march value and compiler flag.
# MARCH (string) wins when non-empty: any value gcc accepts ("x86-64-v3",
# "znver4", "native"), or "none" for no arch flag. An empty MARCH falls back to
# the legacy NATIVE bool: ON => "native", OFF => "none". Sets <out_march> to the
# resolved value and <out_flag> to "-march=<value>" or "" for "none".
macro(diffrg_resolve_march out_march out_flag)
  if(NOT MARCH STREQUAL "")
    set(${out_march} "${MARCH}")
    if(NOT NATIVE)
      message(
        "  ${BoldYellow}[arch] Both -DMARCH=${MARCH} and -DNATIVE=OFF given; MARCH wins.${ColourReset}"
      )
    endif()
  elseif(NATIVE)
    set(${out_march} "native")
  else()
    set(${out_march} "none")
  endif()
  if(${out_march} STREQUAL "none")
    set(${out_flag} "")
  else()
    set(${out_flag} "-march=${${out_march}}")
  endif()
endmacro()
