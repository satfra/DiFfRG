# add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)

# boost_json.cc includes boost/json/src.hpp (the compiled boost::json source).
# It must NOT be compiled with CUDA offloading, otherwise the static member
# key_value_pair::empty_ ends up in device code and causes nvlink duplicate
# symbol errors when libDiFfRG.a appears more than once on the link line.
add_library(DiFfRG_boost_json STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/boost_json.cc)
set_target_properties(DiFfRG_boost_json PROPERTIES
    LANGUAGE CXX
    LINKER_LANGUAGE CXX
    CUDA_SEPARABLE_COMPILATION OFF)
target_include_directories(DiFfRG_boost_json SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})

# toml_json.cc is the only translation unit that sees toml11, an 18k-line header-only
# library. Like boost_json.cc it is kept out of the CUDA path: pushing it through
# nvcc/nvcc_wrapper is slow and risks the same duplicate-symbol trouble.
add_library(DiFfRG_toml STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/toml_json.cc)
set_target_properties(DiFfRG_toml PROPERTIES
    LANGUAGE CXX
    LINKER_LANGUAGE CXX
    CUDA_SEPARABLE_COMPILATION OFF)
target_include_directories(DiFfRG_toml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(DiFfRG_toml SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})

set(DiFfRG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/configuration_helper.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/csv.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/csv_reader.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/external_data_interpolator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/init.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/interpolation.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/config_tree.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/eigen.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/mpi.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/polynomials.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/run_reporter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/csv_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/diagnostic_port.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/fe_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/hdf5_frame.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/hdf5_writer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/hdf5_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/hdf5_input.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/output_path.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/output_settings.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/output_session.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/output_timings.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/snapshot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/common/eom_config.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/mesh/rectangular_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/model/model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/solver/kinsol.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/solver/newton.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/boost_rk.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/boost_abm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/explicit_euler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/implicit_euler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/rk.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_arkode.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida_boost_rk.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida_boost_abm.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida_arkode.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/trbdf2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/integration/abstract_integrator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/integration/map_scheduler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/matsubara.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/quadrature_provider.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/quadrature.cc)

add_library(DiFfRG STATIC ${DiFfRG_SOURCES})

target_link_libraries(DiFfRG PRIVATE DiFfRG_boost_json)
target_link_libraries(DiFfRG PRIVATE DiFfRG_toml)

setup_target(DiFfRG)

target_include_directories(DiFfRG PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>)
message(STATUS "HDF5 (static) support enabled")

add_library(DiFfRG::DiFfRG ALIAS DiFfRG)
set_property(TARGET DiFfRG PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

include(GNUInstallDirs)
install(
  TARGETS DiFfRG DiFfRG_boost_json DiFfRG_toml
  EXPORT DiFfRGTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
