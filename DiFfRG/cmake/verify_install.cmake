# ==============================================================================
# DiFfRG Installation Verification Script
# ==============================================================================
#
# Usage:
#   cmake -DBUNDLED_DIR=~/.local/share/DiFfRG/bundled -P verify_install.cmake
#
# This script checks that all required DiFfRG dependencies can be found
# by looking for their CMake config files in the BUNDLED_DIR.
# ==============================================================================

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED BUNDLED_DIR)
  message(FATAL_ERROR "BUNDLED_DIR must be set. Usage:\n  cmake -DBUNDLED_DIR=<path> -P verify_install.cmake")
endif()

# Color codes
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(Green "${Esc}[32m")
  set(Red "${Esc}[31m")
  set(Yellow "${Esc}[33m")
  set(Bold "${Esc}[1m")
  set(Reset "${Esc}[m")
endif()

set(_all_passed TRUE)

# Check for a dependency by looking for its CMake config file
#
# Pass ALLOW_SYSTEM to additionally accept the dependency when it is provided by
# a system install (outside BUNDLED_DIR). Used for Boost, TBB and HDF5, which
# DiFfRG may consume from the system instead of the bundled build (see the
# <LIB>_DIR / BUILD_<LIB> options in superbuild.cmake).
macro(verify_dep name)
  cmake_parse_arguments(_VD "ALLOW_SYSTEM" "" "CONFIG_NAMES" ${ARGN})

  # Pad name to 15 chars for alignment
  string(LENGTH "${name}" _name_len)
  math(EXPR _pad "15 - ${_name_len}")
  if(_pad LESS 0)
    set(_pad 0)
  endif()
  string(REPEAT " " ${_pad} _spaces)

  set(_found FALSE)
  set(_found_path "")

  # Search for config files
  if(NOT DEFINED _VD_CONFIG_NAMES)
    set(_VD_CONFIG_NAMES "${name}Config.cmake" "${name}-config.cmake"
      "${name}/${name}Config.cmake" "${name}/${name}-config.cmake")
  endif()

  foreach(_search_dir "${BUNDLED_DIR}" "${BUNDLED_DIR}/lib/cmake" "${BUNDLED_DIR}/lib64/cmake"
                      "${BUNDLED_DIR}/share/cmake" "${BUNDLED_DIR}/share")
    if(NOT _found AND EXISTS "${_search_dir}")
      foreach(_config_name ${_VD_CONFIG_NAMES})
        if(NOT _found)
          file(GLOB_RECURSE _matches "${_search_dir}/${_config_name}")
          if(_matches)
            list(GET _matches 0 _found_path)
            set(_found TRUE)
          endif()
        endif()
      endforeach()
    endif()
  endforeach()

  # Fall back to a system install when explicitly allowed (e.g. system Boost).
  if(NOT _found AND _VD_ALLOW_SYSTEM)
    foreach(_sys_dir "/usr/lib/cmake" "/usr/lib64/cmake" "/usr/local/lib/cmake"
                     "/usr/share/cmake" "/opt/homebrew/lib/cmake")
      if(NOT _found AND EXISTS "${_sys_dir}")
        foreach(_config_name ${_VD_CONFIG_NAMES})
          if(NOT _found)
            file(GLOB_RECURSE _sys_matches "${_sys_dir}/${_config_name}")
            if(_sys_matches)
              list(GET _sys_matches 0 _found_path)
              set(_found TRUE)
            endif()
          endif()
        endforeach()
      endif()
    endforeach()
  endif()

  if(_found)
    message("  ${Green}PASS${Reset}  ${name}${_spaces}  ${_found_path}")
  else()
    message("  ${Red}FAIL${Reset}  ${name}${_spaces}  NOT FOUND")
    set(_all_passed FALSE)
  endif()
endmacro()

# Check for a system dependency by looking for it in standard paths
macro(verify_system_dep name pkg_hint)
  string(LENGTH "${name}" _name_len)
  math(EXPR _pad "15 - ${_name_len}")
  if(_pad LESS 0)
    set(_pad 0)
  endif()
  string(REPEAT " " ${_pad} _spaces)

  # Look for the cmake config in standard system paths
  set(_sys_found FALSE)
  foreach(_sys_dir "/usr/lib/cmake" "/usr/lib64/cmake" "/usr/local/lib/cmake"
                   "/usr/share/cmake" "/opt/homebrew/lib/cmake")
    if(NOT _sys_found AND EXISTS "${_sys_dir}")
      file(GLOB _sys_matches "${_sys_dir}/${name}*/${name}Config.cmake"
                              "${_sys_dir}/${name}*/${name}-config.cmake"
                              "${_sys_dir}/${name}*config.cmake")
      if(_sys_matches)
        set(_sys_found TRUE)
      endif()
    endif()
  endforeach()

  # Also check for pkg-config
  if(NOT _sys_found)
    find_program(_pkg_config pkg-config)
    if(_pkg_config)
      execute_process(COMMAND ${_pkg_config} --exists ${pkg_hint}
                      RESULT_VARIABLE _pkg_result)
      if(_pkg_result EQUAL 0)
        set(_sys_found TRUE)
      endif()
    endif()
  endif()

  if(_sys_found)
    message("  ${Green}PASS${Reset}  ${name}${_spaces}  found (system)")
  else()
    message("  ${Red}FAIL${Reset}  ${name}${_spaces}  NOT FOUND (system dependency)")
    set(_all_passed FALSE)
  endif()
endmacro()

message("")
message("${Bold}======================================================================${Reset}")
message("${Bold}  DiFfRG Installation Verification${Reset}")
message("${Bold}======================================================================${Reset}")
message("  BUNDLED_DIR: ${BUNDLED_DIR}")
message("----------------------------------------------------------------------")

# Bundled dependencies
verify_dep(deal.II  CONFIG_NAMES "deal.IIConfig.cmake")
verify_dep(TBB      ALLOW_SYSTEM CONFIG_NAMES "TBBConfig.cmake" "tbb/TBBConfig.cmake")
verify_dep(Kokkos   CONFIG_NAMES "KokkosConfig.cmake")
verify_dep(Boost    ALLOW_SYSTEM CONFIG_NAMES "BoostConfig.cmake" "boost_headers/BoostConfig.cmake")
verify_dep(Eigen3   CONFIG_NAMES "Eigen3Config.cmake")
verify_dep(autodiff CONFIG_NAMES "autodiffConfig.cmake")
verify_dep(spdlog   CONFIG_NAMES "spdlogConfig.cmake")
verify_dep(HDF5     ALLOW_SYSTEM CONFIG_NAMES "hdf5-config.cmake" "HDF5Config.cmake")

# System dependencies
verify_system_dep(GSL "gsl")

message("----------------------------------------------------------------------")
if(_all_passed)
  message("  ${Green}${Bold}All required dependencies found.${Reset}")
else()
  message("  ${Red}${Bold}Some dependencies are missing. See above for details.${Reset}")
  message("")
  message("  To build missing bundled dependencies, run from the repo root:")
  message("    mkdir build && cd build")
  message("    cmake .. -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG")
  message("    cmake --build . -- -j8")
  message("")
  message("  For system dependencies (GSL), install via your package manager:")
  message("    Ubuntu/Debian:  sudo apt install libgsl-dev")
  message("    Arch Linux:     sudo pacman -S gsl")
  message("    Rocky/RHEL:     sudo dnf install gsl-devel")
  message("    macOS:          brew install gsl")
endif()
message("${Bold}======================================================================${Reset}")
message("")
