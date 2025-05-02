cmake_minimum_required(VERSION 3.27)

project(DiFfRG)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# get the user's home directory
set(HOME_DIR $ENV{HOME})

# if the install prefix is not set, set DEPEND_INSTALL_PREFIX to the build directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  message(WARNING "CMAKE_INSTALL_PREFIX not set. Setting to ${HOME_DIR}/.local/share/DiFfRG")
  set(CMAKE_INSTALL_PREFIX ${HOME_DIR}/.local/share/DiFfRG CACHE PATH "Install prefix" FORCE)
endif()
set(DEPEND_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/bundled/ CACHE PATH "Install prefix for dependencies")

message(STATUS "Dependencies will be installed to ${DEPEND_INSTALL_PREFIX}")

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Set up external dependencies

include (ExternalProject)

# build DiFfRG dependencies
ExternalProject_Add (DiFfRG_dependencies
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/dependencies
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS
  -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
)

# build deal.II
ExternalProject_Add (deal.II_dep
  GIT_REPOSITORY https://github.com/dealii/dealii.git
  GIT_TAG        v9.6.2
	PREFIX ${CMAKE_CURRENT_BINARY_DIR}/deal.II
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS DiFfRG_dependencies
  BUILD_COMMAND make
  INSTALL_COMMAND make install
	CMAKE_CACHE_ARGS
  -DENABLE_MPI:BOOL=OFF
  -DCMAKE_BUILD_TYPE:STRING=DebugRelease
  -DDEAL_II_COMPONENT_EXAMPLES:BOOL=OFF
  -DDEAL_II_COMPONENT_DOCUMENTATION:BOOL=OFF
  -DDEAL_II_WITH_MPI:BOOL=OFF
  -DDEAL_II_WITH_UMFPACK:BOOL=ON
  -DDEAL_II_WITH_TASKFLOW:BOOL=OFF
  -DDEAL_II_WITH_VTK:BOOL=OFF
  -DCMAKE_CXX_STANDARD:STRING=17
  -DKOKKOS_DIR:STRING=${DEPEND_INSTALL_PREFIX}
  -DTBB_DIR:STRING=${DEPEND_INSTALL_PREFIX}
  -DSUNDIALS_DIR:STRING=${DEPEND_INSTALL_PREFIX}
  -DBOOST_DIR:STRING=${DEPEND_INSTALL_PREFIX}
  -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
  )

# build the actual DiFfRG library
ExternalProject_Add (DiFfRG
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/
	PREFIX ${CMAKE_CURRENT_BINARY_DIR}/DiFfRG/
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  DEPENDS DiFfRG_dependencies deal.II_dep
  BUILD_COMMAND make
  INSTALL_COMMAND make install
	CMAKE_CACHE_ARGS
  -DENABLE_MPI:BOOL=OFF
  -DBUNDLED_DIR:PATH=${DEPEND_INSTALL_PREFIX}
  )