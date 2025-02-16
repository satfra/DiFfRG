# add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)
set(DiFfRG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/boost_json.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/configuration_helper.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/csv_reader.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/external_data_interpolator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/interpolation.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/json.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/math.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/polynomials.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/csv_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/data_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/fe_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/mesh/rectangular_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/model/model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/solver/kinsol.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/solver/newton.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/boost_rk.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/boost_abm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/explicit_euler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/implicit_euler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/rk.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_arkode.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida_boost_rk.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida_boost_abm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida_arkode.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/trbdf2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/flow_equations.cc)
if(USE_CUDA)
  list(
    APPEND
    DiFfRG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/cuda_prefix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/matsubara.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/quadrature_provider.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/quadrature.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/linear_interpolation_1D.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/linear_interpolation_2D.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/linear_interpolation_3D.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/tex_linear_interpolation_1D.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/tex_linear_interpolation_2D.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/tex_linear_interpolation_3D.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/tex_linear_interpolation_1D_stack.cu
  )
else()
  list(
    APPEND
    DiFfRG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/matsubara.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/quadrature_provider.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/quadrature.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/linear_interpolation_1D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/linear_interpolation_2D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/linear_interpolation_3D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/tex_linear_interpolation_1D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/tex_linear_interpolation_2D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/tex_linear_interpolation_3D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/interpolation/tex_linear_interpolation_1D_stack.cc
  )
endif()

add_library(DiFfRG STATIC ${DiFfRG_SOURCES})

setup_target(DiFfRG)

target_include_directories(
  DiFfRG INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                   $<INSTALL_INTERFACE:include>)
add_library(DiFfRG::DiFfRG ALIAS DiFfRG)
set_property(TARGET DiFfRG PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

include(GNUInstallDirs)
install(
  TARGETS DiFfRG
  EXPORT DiFfRGTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
