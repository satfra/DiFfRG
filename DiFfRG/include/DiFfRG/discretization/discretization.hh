#pragma once

#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>

// Finite volume (Kurganov-Tadmor). The discretization pulls in the KT assembler; the
// reconstructors, limiters and wave speeds that parameterize it are listed explicitly so that
// every choice a user can make is available from the umbrella header.
#include <DiFfRG/discretization/FV/discretization.hh>

#include <DiFfRG/discretization/FV/limiter/central_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/superbee_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/van_albada_limiter.hh>

#include <DiFfRG/discretization/FV/reconstructor/advection/first_order_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/diffusion/corrected_weighted_least_squares_reconstructor.hh>

#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>

#include <DiFfRG/discretization/variables/variables.hh>

#include <DiFfRG/discretization/common/eom_config.hh>

#include <DiFfRG/discretization/data/csv_output.hh>
#include <DiFfRG/discretization/data/data.hh>
// Tombstone for the removed DataOutput; kept reachable from the umbrella header so that
// pre-OutputSession code gets the guided migration error instead of "no such type".
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/data/output_settings.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/data/hdf5_input.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>

#include <DiFfRG/discretization/mesh/h_adaptivity.hh>
#include <DiFfRG/discretization/mesh/no_adaptivity.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

#include <DiFfRG/discretization/coordinates/combined_coordinates.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/discretization/coordinates/stack_coordinates.hh>
