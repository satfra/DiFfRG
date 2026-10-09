#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/model/model.hh"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <algorithm>
#include <autodiff/forward/real.hpp>
#include <boilerplate/kt_models.hh>
#include <boilerplate/kt_trace_evaluation.hh>
#include <cmath>
#include <cstddef>
#include <deal.II/base/numbers.h>
#include <deal.II/lac/vector.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tuple>
#include <utility>
#include <vector>

// #include <petscvec.h>

using NumberType = double;
using VectorType = dealii::Vector<NumberType>;
using namespace dealii;
namespace KT = DiFfRG::FV::KurganovTadmor;

/// Cell width handed to the flux helpers below. None of the test models reads it; it is here so the
/// calls match the signature that carries the local grid spacing through to a model's flux.
static constexpr double test_cell_width = 1.0;

/// Empty extractor / variable data for the direct helper calls below. The models used there declare
/// neither, so their fluxes never read them; they are here because the helpers take them
/// unconditionally.
static const std::array<double, 0> no_extractors{};
static const VectorType no_variables{};
using WaveSpeedStrategy = DiFfRG::FV::KurganovTadmor::MaxEigenvalueWaveSpeed;
using DiFfRG::Testing::compute_diffusion_flux;
using DiFfRG::Testing::compute_diffusion_flux_jacobian;
using KT::internal::compute_numerical_flux;
using KT::internal::reconstruct_u;

template <typename Assembler>
concept CanMakeAssemblyContextViewFromTemporaryCache = requires(const Assembler &assembler) {
  assembler.make_assembly_context_view(typename Assembler::SolutionReconstructionCache{});
};

using FEFunctionDesc = DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u">>;
using Components = DiFfRG::ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

using CurlFreeFEFunctionDesc = DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u">, DiFfRG::Scalar<"v">>;
using CurlFreeComponents = DiFfRG::ComponentDescriptor<CurlFreeFEFunctionDesc>;

class TestModel : public DiFfRG::def::AbstractModel<TestModel, Components>
{
public:
  template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
  static void flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                   [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol)
  {
    auto u = get<"fe_functions">(sol);
    F_i[idxf("u")][0] = u[0] * u[0] / 2.0 + x[0];
  }
};

class NonlinearDiffusionProbeModel
{
public:
  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution &sol) const
  {
    const auto &du = get<"fe_derivatives">(sol);
    F_i[0][0] = du[0][1] * du[0][1];
    F_i[0][1] = NT(0.0);
  }
};

class ResidualContributionModel : public DiFfRG::def::AbstractModel<ResidualContributionModel, Components>,
                                  public DiFfRG::def::Time,
                                  public DiFfRG::def::LLFFlux<ResidualContributionModel>,
                                  public DiFfRG::def::FlowBoundaries<ResidualContributionModel>,
                                  public DiFfRG::def::FVDefaultBoundaries<ResidualContributionModel>,
                                  public DiFfRG::def::AD<ResidualContributionModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.25 * pos[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const { return {1.0 + 0.25 * pos[0]}; }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    F_i[0][0] = 0.5 * u[0] * u[0];
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &du = get<"fe_derivatives">(sol);
    F_i[0][0] = 0.1 * du[0][0];
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> &pos, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    s_i[0] = 0.5 + pos[0] + 0.25 * u[0];
  }

  template <typename NT, typename Vector, typename VectorDot>
  void mass(std::array<NT, 1> &m_i, const Point<1> & /*pos*/, const Vector &u, const VectorDot &dt_u) const
  {
    m_i[0] = dt_u[0] + 0.125 * u[0];
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(DiFfRG::def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              DiFfRG::def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    DiFfRG::Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                               [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }
};

class FaceHookProbeModel : public DiFfRG::def::AbstractModel<FaceHookProbeModel, Components>,
                           public DiFfRG::def::Time,
                           public DiFfRG::def::LLFFlux<FaceHookProbeModel>,
                           public DiFfRG::def::FlowBoundaries<FaceHookProbeModel>,
                           public DiFfRG::def::FVDefaultBoundaries<FaceHookProbeModel>,
                           public DiFfRG::def::AD<FaceHookProbeModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.5 * pos[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const { return {1.0 + 0.5 * pos[0]}; }

  template <KT::HasAssemblyContextView Context>
  void fv_kt_pre_assembly(const KT::AssemblyStage stage, const Context &context)
  {
    ++hook_calls;
    hook_stage = stage;
    hook_face_count = 0;
    hook_cell_count = 0;
    saw_boundary_face = false;
    saw_interior_face = false;
    hook_ready = false;

    hook_min_u_plus = std::numeric_limits<double>::infinity();
    for (const auto face : context.faces()) {
      ++hook_face_count;
      saw_boundary_face = saw_boundary_face || face.at_boundary();
      saw_interior_face = saw_interior_face || !face.at_boundary();
      const auto &reconstruction = face.reconstruction();
      last_u_minus = reconstruction.u_minus[0];
      last_u_plus = reconstruction.u_plus[0];
      last_grad_minus = reconstruction.face_grad_minus[0][0];
      last_grad_plus = reconstruction.face_grad_plus[0][0];
      hook_min_u_plus = std::min(hook_min_u_plus, static_cast<double>(reconstruction.u_plus[0]));
    }

    const auto &faces = context.faces();
    hook_parallel_min_u_plus = tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, faces.size()), std::numeric_limits<double>::infinity(),
        [&](const tbb::blocked_range<std::size_t> &range, double local_min) {
          for (std::size_t i = range.begin(); i != range.end(); ++i) {
            const auto face = faces[i];
            local_min = std::min(local_min, static_cast<double>(face.reconstruction().u_plus[0]));
          }
          return local_min;
        },
        [](const double left, const double right) { return std::min(left, right); });

    hook_min_cell_u = std::numeric_limits<double>::infinity();
    const auto &cells = context.cells();
    for (std::size_t i = 0; i < cells.size(); ++i) {
      const auto cell = cells[i];
      ++hook_cell_count;
      last_cell_u = cell.stencil().cell.u[0];
      last_cell_point = cell.point()[0];
      hook_min_cell_u = std::min(hook_min_cell_u, static_cast<double>(cell.stencil().cell.u[0]));
    }

    hook_flux_scale = hook_min_u_plus;
    hook_ready = true;
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> &pos, const Solution &sol) const
  {
    const auto &du = get<"fe_derivatives">(sol);
    const double gradient = static_cast<double>(autodiff::detail::val(du[0][0]));
    kt_flux_gradients_match = kt_flux_gradients_match && std::abs(gradient - 0.5) < 1.0e-12;
    if (std::abs(pos[0]) < 1.0e-12 || std::abs(pos[0] - 1.0) < 1.0e-12)
      kt_flux_saw_boundary_gradient = true;
    else
      kt_flux_saw_interior_gradient = true;
    F_i[0][0] = 0.0;
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &du = get<"fe_derivatives">(sol);
    ++flux_calls;
    flux_saw_hook_state = flux_saw_hook_state || hook_ready;
    F_i[0][0] = NT(hook_flux_scale) * du[0][0];
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(DiFfRG::def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              DiFfRG::def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    DiFfRG::Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                               [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }

  int hook_calls = 0;
  KT::AssemblyStage hook_stage = KT::AssemblyStage::residual;
  unsigned int hook_face_count = 0;
  unsigned int hook_cell_count = 0;
  bool saw_boundary_face = false;
  bool saw_interior_face = false;
  bool hook_ready = false;
  double hook_min_u_plus = std::numeric_limits<double>::infinity();
  double hook_parallel_min_u_plus = std::numeric_limits<double>::infinity();
  double hook_min_cell_u = std::numeric_limits<double>::infinity();
  double hook_flux_scale = -1.0;
  double last_u_minus = std::numeric_limits<double>::quiet_NaN();
  double last_u_plus = std::numeric_limits<double>::quiet_NaN();
  double last_grad_minus = std::numeric_limits<double>::quiet_NaN();
  double last_grad_plus = std::numeric_limits<double>::quiet_NaN();
  double last_cell_u = std::numeric_limits<double>::quiet_NaN();
  double last_cell_point = std::numeric_limits<double>::quiet_NaN();
  mutable int flux_calls = 0;
  mutable bool flux_saw_hook_state = false;
  mutable bool kt_flux_gradients_match = true;
  mutable bool kt_flux_saw_boundary_gradient = false;
  mutable bool kt_flux_saw_interior_gradient = false;
};

class ThirdDerivativeFluxProbeModel : public DiFfRG::def::AbstractModel<ThirdDerivativeFluxProbeModel, Components>,
                                      public DiFfRG::def::Time,
                                      public DiFfRG::def::LLFFlux<ThirdDerivativeFluxProbeModel>,
                                      public DiFfRG::def::FlowBoundaries<ThirdDerivativeFluxProbeModel>,
                                      public DiFfRG::def::FVDefaultBoundaries<ThirdDerivativeFluxProbeModel>,
                                      public DiFfRG::def::AD<ThirdDerivativeFluxProbeModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = solution(pos)[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const
  {
    const double x = pos[0];
    return {1.0 + 0.5 * x - 0.25 * x * x + x * x * x};
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = 0.0;
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &third_derivatives = get<"fe_third_derivatives">(sol);
    const auto third = third_derivatives[0][0][0][0];
    saw_third_derivative = true;
    min_third_derivative = std::min(min_third_derivative, static_cast<double>(third));
    max_third_derivative = std::max(max_third_derivative, static_cast<double>(third));
    saw_exact_cubic_third_derivative =
        saw_exact_cubic_third_derivative || std::abs(static_cast<double>(third) - 6.0) < 1e-10;
    F_i[0][0] = third;
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(DiFfRG::def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              DiFfRG::def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    DiFfRG::Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                               [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }

  mutable bool saw_third_derivative = false;
  mutable bool saw_exact_cubic_third_derivative = false;
  mutable double min_third_derivative = std::numeric_limits<double>::infinity();
  mutable double max_third_derivative = -std::numeric_limits<double>::infinity();
};

class NamedSlotAffineBoundary2DModel : public DiFfRG::def::AbstractModel<NamedSlotAffineBoundary2DModel, Components>,
                                       public DiFfRG::def::Time,
                                       public DiFfRG::def::LLFFlux<NamedSlotAffineBoundary2DModel>,
                                       public DiFfRG::def::FlowBoundaries<NamedSlotAffineBoundary2DModel>,
                                       public DiFfRG::def::FVDefaultBoundaries<NamedSlotAffineBoundary2DModel>,
                                       public DiFfRG::def::AD<NamedSlotAffineBoundary2DModel>
{
public:
  static constexpr double offset = 1.25;
  static constexpr double x_slope = 0.35;
  static constexpr double y_slope = -0.2;

  template <typename Vector> void initial_condition(const Point<2> &pos, Vector &values) const
  {
    values[0] = solution(pos)[0];
  }

  std::array<double, 1> solution(const Point<2> &pos) const { return {offset + x_slope * pos[0] + y_slope * pos[1]}; }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    s_i[0] = NT(0.0);
  }
};

class OriginOddCurlFreeBoundary2DModel
    : public DiFfRG::def::AbstractModel<OriginOddCurlFreeBoundary2DModel, CurlFreeComponents>,
      public DiFfRG::def::Time,
      public DiFfRG::def::LLFFlux<OriginOddCurlFreeBoundary2DModel>,
      public DiFfRG::def::FlowBoundaries<OriginOddCurlFreeBoundary2DModel>,
      public DiFfRG::def::OriginOddLinearExtrapolationBoundaries<OriginOddCurlFreeBoundary2DModel>,
      public DiFfRG::def::AD<OriginOddCurlFreeBoundary2DModel>
{
public:
  static constexpr double x_slope = 0.75;
  static constexpr double y_slope = -0.5;

  template <typename Vector> void initial_condition(const Point<2> &pos, Vector &values) const
  {
    const auto values_at_pos = solution(pos);
    values[0] = values_at_pos[0];
    values[1] = values_at_pos[1];
  }

  std::array<double, 2> solution(const Point<2> &pos) const { return {x_slope * pos[0], y_slope * pos[1]}; }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 2, NT>, 2> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    for (auto &flux_i : F_i)
      flux_i = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 2, NT>, 2> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    for (auto &flux_i : F_i)
      flux_i = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 2> &s_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    for (auto &source_i : s_i)
      source_i = NT(0.0);
  }
};

class NonAffineTangentialGhostBoundary2DModel
    : public DiFfRG::def::AbstractModel<NonAffineTangentialGhostBoundary2DModel, Components>,
      public DiFfRG::def::Time,
      public DiFfRG::def::LLFFlux<NonAffineTangentialGhostBoundary2DModel>,
      public DiFfRG::def::FlowBoundaries<NonAffineTangentialGhostBoundary2DModel>,
      public DiFfRG::def::AD<NonAffineTangentialGhostBoundary2DModel>
{
public:
  static constexpr double offset = 1.0;
  static constexpr double x_slope = 0.5;
  static constexpr double y_slope = -0.25;
  static constexpr double lower_shift = 3.0;
  static constexpr double upper_shift = -2.0;
  static constexpr double tangential_shift_slope = 0.75;

  template <typename Vector> void initial_condition(const Point<2> &pos, Vector &values) const
  {
    values[0] = solution(pos)[0];
  }

  std::array<double, 1> solution(const Point<2> &pos) const { return {offset + x_slope * pos[0] + y_slope * pos[1]}; }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    s_i[0] = NT(0.0);
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(DiFfRG::def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              DiFfRG::def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 2);
    static_assert(n_components == 1);
    using namespace DiFfRG::def::BoundaryStencilIndex;

    unsigned int axis = 0;
    const double distance_0 = std::abs(x_face[0] - x_stencil[physical_cell][0]);
    const double distance_1 = std::abs(x_face[1] - x_stencil[physical_cell][1]);
    if (distance_1 > distance_0) axis = 1;

    const bool lower_boundary = x_face[axis] <= x_stencil[physical_cell][axis];
    const double delta = lower_boundary ? (x_stencil[upper_inner][axis] - x_stencil[physical_cell][axis])
                                        : (x_stencil[physical_cell][axis] - x_stencil[lower_inner][axis]);
    const unsigned int tangential_axis = 1U - axis;
    const NT shift = (lower_boundary ? NT(lower_shift) : NT(upper_shift)) +
                     NT(tangential_shift_slope) * NT(x_stencil[physical_cell][tangential_axis]);
    if (lower_boundary) {
      x_stencil[lower_inner] = x_stencil[physical_cell];
      x_stencil[lower_outer] = x_stencil[physical_cell];
      x_stencil[lower_inner][axis] = x_stencil[physical_cell][axis] - delta;
      x_stencil[lower_outer][axis] = x_stencil[physical_cell][axis] - 2.0 * delta;
      u_stencil[lower_inner][0] = NT(2.0) * u_stencil[physical_cell][0] - u_stencil[upper_inner][0] + shift;
      u_stencil[lower_outer][0] =
          NT(3.0) * u_stencil[physical_cell][0] - NT(2.0) * u_stencil[upper_inner][0] + NT(2.0) * shift;
      return true;
    }

    x_stencil[upper_inner] = x_stencil[physical_cell];
    x_stencil[upper_outer] = x_stencil[physical_cell];
    x_stencil[upper_inner][axis] = x_stencil[physical_cell][axis] + delta;
    x_stencil[upper_outer][axis] = x_stencil[physical_cell][axis] + 2.0 * delta;
    u_stencil[upper_inner][0] = NT(2.0) * u_stencil[physical_cell][0] - u_stencil[lower_inner][0] + shift;
    u_stencil[upper_outer][0] =
        NT(3.0) * u_stencil[physical_cell][0] - NT(2.0) * u_stencil[lower_inner][0] + NT(2.0) * shift;
    return true;
  }
};

static DiFfRG::ConfigTree make_fv_test_json()
{
  return DiFfRG::json::value(
      {{"physical", {{"Lambda", 1.}}},
       {"discretization",
        {{"fe_order", 0},
         {"overintegration", 0},
         {"output_subdivisions", 1},
         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},
         {"grid", {{"x_grid", "0:0.25:1"}, {"y_grid", "0:1:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
       {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", true}}}});
}

static DiFfRG::ConfigTree make_fv_eom_potential_output_json()
{
  auto json = make_fv_test_json();
  json.set_uint("/discretization/EoM_max_iter", 100);
  json.set_string("/discretization/grid/x_grid", "0:0.1:1");
  json.set_bool("/output/vtk", true);
  json.set_bool("/output/hdf5", false);
  return json;
}

static void ensure_logger()
{
  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
  } catch (const spdlog::spdlog_ex &) {
  }
}

static std::string read_text_file(const std::filesystem::path &path)
{
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

template <typename State> static void check_face_reconstruction_state_1d(const State &actual, const State &expected)
{
  REQUIRE(actual.u_minus.size() == expected.u_minus.size());
  for (std::size_t c = 0; c < actual.u_minus.size(); ++c) {
    CHECK(actual.u_minus[c] == Catch::Approx(expected.u_minus[c]).margin(1e-14));
    CHECK(actual.u_plus[c] == Catch::Approx(expected.u_plus[c]).margin(1e-14));
    CHECK(actual.center_grad_minus[c][0] == Catch::Approx(expected.center_grad_minus[c][0]).margin(1e-14));
    CHECK(actual.center_grad_plus[c][0] == Catch::Approx(expected.center_grad_plus[c][0]).margin(1e-14));
    CHECK(actual.face_grad_minus[c][0] == Catch::Approx(expected.face_grad_minus[c][0]).margin(1e-14));
    CHECK(actual.face_grad_plus[c][0] == Catch::Approx(expected.face_grad_plus[c][0]).margin(1e-14));
    CHECK(actual.diffusion_u_minus[c] == Catch::Approx(expected.diffusion_u_minus[c]).margin(1e-14));
    CHECK(actual.diffusion_u_plus[c] == Catch::Approx(expected.diffusion_u_plus[c]).margin(1e-14));
    CHECK(actual.diffusion_grad_minus[c][0] == Catch::Approx(expected.diffusion_grad_minus[c][0]).margin(1e-14));
    CHECK(actual.diffusion_grad_plus[c][0] == Catch::Approx(expected.diffusion_grad_plus[c][0]).margin(1e-14));
    CHECK(actual.diffusion_u[c] == Catch::Approx(expected.diffusion_u[c]).margin(1e-14));
    CHECK(actual.diffusion_grad[c][0] == Catch::Approx(expected.diffusion_grad[c][0]).margin(1e-14));
  }
}

template <typename State> static void check_face_reconstruction_state_reversed_1d(const State &left, const State &right)
{
  REQUIRE(left.u_minus.size() == right.u_minus.size());
  for (std::size_t c = 0; c < left.u_minus.size(); ++c) {
    CHECK(left.u_minus[c] == Catch::Approx(right.u_plus[c]).margin(1e-14));
    CHECK(left.u_plus[c] == Catch::Approx(right.u_minus[c]).margin(1e-14));
    CHECK(left.center_grad_minus[c][0] == Catch::Approx(right.center_grad_plus[c][0]).margin(1e-14));
    CHECK(left.center_grad_plus[c][0] == Catch::Approx(right.center_grad_minus[c][0]).margin(1e-14));
    CHECK(left.face_grad_minus[c][0] == Catch::Approx(right.face_grad_plus[c][0]).margin(1e-14));
    CHECK(left.face_grad_plus[c][0] == Catch::Approx(right.face_grad_minus[c][0]).margin(1e-14));
    CHECK(left.diffusion_u_minus[c] == Catch::Approx(right.diffusion_u_plus[c]).margin(1e-14));
    CHECK(left.diffusion_u_plus[c] == Catch::Approx(right.diffusion_u_minus[c]).margin(1e-14));
    CHECK(left.diffusion_grad_minus[c][0] == Catch::Approx(right.diffusion_grad_plus[c][0]).margin(1e-14));
    CHECK(left.diffusion_grad_plus[c][0] == Catch::Approx(right.diffusion_grad_minus[c][0]).margin(1e-14));
    CHECK(left.diffusion_u[c] == Catch::Approx(right.diffusion_u[c]).margin(1e-14));
    CHECK(left.diffusion_grad[c][0] == Catch::Approx(right.diffusion_grad[c][0]).margin(1e-14));
  }
}

template <int dim, typename State>
static void check_face_reconstruction_state(const State &actual, const State &expected)
{
  REQUIRE(actual.u_minus.size() == expected.u_minus.size());
  for (std::size_t c = 0; c < actual.u_minus.size(); ++c) {
    CHECK(actual.u_minus[c] == Catch::Approx(expected.u_minus[c]).margin(1e-14));
    CHECK(actual.u_plus[c] == Catch::Approx(expected.u_plus[c]).margin(1e-14));
    for (unsigned int d = 0; d < dim; ++d) {
      CHECK(actual.center_grad_minus[c][d] == Catch::Approx(expected.center_grad_minus[c][d]).margin(1e-14));
      CHECK(actual.center_grad_plus[c][d] == Catch::Approx(expected.center_grad_plus[c][d]).margin(1e-14));
      CHECK(actual.face_grad_minus[c][d] == Catch::Approx(expected.face_grad_minus[c][d]).margin(1e-14));
      CHECK(actual.face_grad_plus[c][d] == Catch::Approx(expected.face_grad_plus[c][d]).margin(1e-14));
      CHECK(actual.diffusion_grad_minus[c][d] == Catch::Approx(expected.diffusion_grad_minus[c][d]).margin(1e-14));
      CHECK(actual.diffusion_grad_plus[c][d] == Catch::Approx(expected.diffusion_grad_plus[c][d]).margin(1e-14));
      CHECK(actual.diffusion_grad[c][d] == Catch::Approx(expected.diffusion_grad[c][d]).margin(1e-14));
    }
    CHECK(actual.diffusion_u_minus[c] == Catch::Approx(expected.diffusion_u_minus[c]).margin(1e-14));
    CHECK(actual.diffusion_u_plus[c] == Catch::Approx(expected.diffusion_u_plus[c]).margin(1e-14));
    CHECK(actual.diffusion_u[c] == Catch::Approx(expected.diffusion_u[c]).margin(1e-14));
  }
}

template <int dim, typename State>
static void check_face_reconstruction_state_reversed(const State &left, const State &right)
{
  REQUIRE(left.u_minus.size() == right.u_minus.size());
  for (std::size_t c = 0; c < left.u_minus.size(); ++c) {
    CHECK(left.u_minus[c] == Catch::Approx(right.u_plus[c]).margin(1e-14));
    CHECK(left.u_plus[c] == Catch::Approx(right.u_minus[c]).margin(1e-14));
    for (unsigned int d = 0; d < dim; ++d) {
      CHECK(left.center_grad_minus[c][d] == Catch::Approx(right.center_grad_plus[c][d]).margin(1e-14));
      CHECK(left.center_grad_plus[c][d] == Catch::Approx(right.center_grad_minus[c][d]).margin(1e-14));
      CHECK(left.face_grad_minus[c][d] == Catch::Approx(right.face_grad_plus[c][d]).margin(1e-14));
      CHECK(left.face_grad_plus[c][d] == Catch::Approx(right.face_grad_minus[c][d]).margin(1e-14));
      CHECK(left.diffusion_grad_minus[c][d] == Catch::Approx(right.diffusion_grad_plus[c][d]).margin(1e-14));
      CHECK(left.diffusion_grad_plus[c][d] == Catch::Approx(right.diffusion_grad_minus[c][d]).margin(1e-14));
      CHECK(left.diffusion_grad[c][d] == Catch::Approx(right.diffusion_grad[c][d]).margin(1e-14));
    }
    CHECK(left.diffusion_u_minus[c] == Catch::Approx(right.diffusion_u_plus[c]).margin(1e-14));
    CHECK(left.diffusion_u_plus[c] == Catch::Approx(right.diffusion_u_minus[c]).margin(1e-14));
    CHECK(left.diffusion_u[c] == Catch::Approx(right.diffusion_u[c]).margin(1e-14));
  }
}

template <typename NT>
static std::pair<KT::internal::CellStencilData<2, NT, 1>, KT::internal::CellStencilData<2, NT, 1>>
make_diffusion_probe_stencils()
{
  using Stencil = KT::internal::CellStencilData<2, NT, 1>;
  Stencil minus{};
  Stencil plus{};
  minus.boundary_ids.fill(dealii::numbers::invalid_boundary_id);
  plus.boundary_ids.fill(dealii::numbers::invalid_boundary_id);
  for (auto &dofs : minus.neighbors.dof_indices)
    dofs.fill(dealii::numbers::invalid_dof_index);
  for (auto &dofs : plus.neighbors.dof_indices)
    dofs.fill(dealii::numbers::invalid_dof_index);
  minus.cell.dof_indices.fill(dealii::numbers::invalid_dof_index);
  plus.cell.dof_indices.fill(dealii::numbers::invalid_dof_index);

  minus.cell.x = Point<2>(0.0, 0.0);
  minus.cell.u = {NT(0.0)};
  minus.neighbors.x = {Point<2>(-1.0, 0.0), Point<2>(1.0, 0.0), Point<2>(0.0, -1.0), Point<2>(0.0, 1.0)};
  minus.neighbors.u = {{{NT(-1.0)}, {NT(3.0)}, {NT(1.0)}, {NT(2.0)}}};

  plus.cell.x = Point<2>(1.0, 0.0);
  plus.cell.u = {NT(3.0)};
  plus.neighbors.x = {Point<2>(0.0, 0.0), Point<2>(2.0, 0.0), Point<2>(1.0, -1.0), Point<2>(1.0, 1.0)};
  plus.neighbors.u = {{{NT(0.0)}, {NT(6.0)}, {NT(4.0)}, {NT(6.0)}}};

  return {minus, plus};
}

TEST_CASE("KT diffusion face gradient is separate from MinMod advective face gradients", "[FV][KT][diffusion][2d]")
{
  using Reconstructor = DiFfRG::def::TVDReconstructor<2, DiFfRG::def::MinModLimiter, double>;
  const auto [minus, plus] = make_diffusion_probe_stencils<double>();

  const Point<2> x_q(0.5, 0.0);
  const auto state = KT::internal::compute_interior_face_reconstruction_state<Reconstructor>(minus, plus, x_q);
  const auto d = plus.cell.x - minus.cell.x;

  CHECK(state.face_grad_minus[0][1] == Catch::Approx(0.0).margin(1.0e-14));
  CHECK(state.face_grad_plus[0][1] == Catch::Approx(0.0).margin(1.0e-14));
  CHECK(state.diffusion_u_minus[0] == Catch::Approx(0.0).margin(1.0e-14));
  CHECK(state.diffusion_u_plus[0] == Catch::Approx(3.0).margin(1.0e-14));
  CHECK(state.diffusion_u[0] == Catch::Approx(1.5).margin(1.0e-14));
  CHECK(state.diffusion_grad_minus[0][0] == Catch::Approx(3.0).margin(1.0e-14));
  CHECK(state.diffusion_grad_plus[0][0] == Catch::Approx(3.0).margin(1.0e-14));
  CHECK(state.diffusion_grad_minus[0][1] == Catch::Approx(0.5).margin(1.0e-14));
  CHECK(state.diffusion_grad_plus[0][1] == Catch::Approx(1.0).margin(1.0e-14));
  CHECK(state.diffusion_grad[0][0] == Catch::Approx(3.0).margin(1.0e-14));
  CHECK(state.diffusion_grad[0][1] == Catch::Approx(0.75).margin(1.0e-14));
  CHECK(dealii::scalar_product(state.diffusion_grad_minus[0], d) ==
        Catch::Approx(plus.cell.u[0] - minus.cell.u[0]).margin(1.0e-14));
  CHECK(dealii::scalar_product(state.diffusion_grad_plus[0], d) ==
        Catch::Approx(plus.cell.u[0] - minus.cell.u[0]).margin(1.0e-14));
}

TEST_CASE("Corrected weighted least-squares diffusion reconstructor preserves AD derivatives",
          "[FV][KT][diffusion][AD][2d]")
{
  using AD = autodiff::Real<1, double>;
  using Reconstructor = DiFfRG::def::CorrectedWeightedLeastSquaresDiffusionReconstructor<2, double>;
  using ADReconstructor = DiFfRG::def::CorrectedWeightedLeastSquaresDiffusionReconstructor<2, AD>;

  auto [minus_ad, plus_ad] = make_diffusion_probe_stencils<AD>();
  seed(minus_ad.neighbors.u[3][0]);
  const auto ad_state = ADReconstructor::template compute_face_state<1>(minus_ad, plus_ad);
  const auto derivatives = Reconstructor::template extract_derivatives<1>(ad_state);
  unseed(minus_ad.neighbors.u[3][0]);

  constexpr double eps = 1.0e-6;
  auto [minus_plus, plus_plus] = make_diffusion_probe_stencils<double>();
  auto [minus_minus, plus_minus] = make_diffusion_probe_stencils<double>();
  minus_plus.neighbors.u[3][0] += eps;
  minus_minus.neighbors.u[3][0] -= eps;
  const auto state_plus = Reconstructor::template compute_face_state<1>(minus_plus, plus_plus);
  const auto state_minus = Reconstructor::template compute_face_state<1>(minus_minus, plus_minus);

  const auto side_u = [](const auto &state, const size_t side) {
    return side == 0 ? state.u_minus[0] : state.u_plus[0];
  };
  const auto side_grad = [](const auto &state, const size_t side, const size_t axis) {
    return side == 0 ? state.grad_minus[0][axis] : state.grad_plus[0][axis];
  };
  const auto finite_difference = [eps](const double plus, const double minus) { return (plus - minus) / (2.0 * eps); };

  for (size_t side = 0; side < 2; ++side) {
    CHECK(derivatives[side].u[0] ==
          Catch::Approx(finite_difference(side_u(state_plus, side), side_u(state_minus, side))).margin(1.0e-8));
    for (size_t axis = 0; axis < 2; ++axis) {
      CHECK(derivatives[side].grad[0][axis] ==
            Catch::Approx(finite_difference(side_grad(state_plus, side, axis), side_grad(state_minus, side, axis)))
                .margin(1.0e-8));
    }
  }
}

TEST_CASE("KT diffusion flux averages nonlinear side fluxes", "[FV][KT][diffusion][2d]")
{
  using Gradient = KT::internal::GradientType<2, double, 1>;
  const NonlinearDiffusionProbeModel model;
  std::array<double, 1> u_minus{{0.0}};
  std::array<double, 1> u_plus{{3.0}};
  Gradient grad_minus{};
  Gradient grad_plus{};
  KT::internal::ThirdDerivativeType<2, double, 1> third_minus{};
  KT::internal::ThirdDerivativeType<2, double, 1> third_plus{};
  grad_minus[0][1] = 0.5;
  grad_plus[0][1] = 1.0;

  const auto D = compute_diffusion_flux<NonlinearDiffusionProbeModel, double, 2, 1>(
      u_minus, u_plus, grad_minus, grad_plus, third_minus, third_plus, Point<2>(), test_cell_width, test_cell_width,
      no_extractors, no_variables, model);
  CHECK(D[0][0] == Catch::Approx(0.625).margin(1.0e-14));

  const auto J = compute_diffusion_flux_jacobian<NonlinearDiffusionProbeModel, double, 2, 1>(
      u_minus, u_plus, grad_minus, grad_plus, third_minus, third_plus, Point<2>(), test_cell_width, test_cell_width,
      no_extractors, no_variables, model);
  CHECK(J.grad[0](0, 0)[0][1] == Catch::Approx(0.5).margin(1.0e-14));
  CHECK(J.grad[1](0, 0)[0][1] == Catch::Approx(1.0).margin(1.0e-14));
}

static DiFfRG::ConfigTree make_kt_boundary_json()
{
  return DiFfRG::json::value(
      {{"physical", {{"Lambda", 1.}}},
       {"discretization",
        {{"fe_order", 0},
         {"overintegration", 0},
         {"output_subdivisions", 1},
         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},
         {"grid", {{"x_grid", "0:0.25:1"}, {"y_grid", "0:0.25:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
       {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", false}}}});
}

template <typename DoFHandler>
static typename DoFHandler::active_cell_iterator find_2d_interior_cell(DoFHandler &dof_handler)
{
  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    bool interior = true;
    for (const auto face_index : cell->face_indices())
      interior = interior && !cell->at_boundary(face_index);
    if (interior) return cell;
  }
  return typename DoFHandler::active_cell_iterator(dof_handler.end());
}

template <typename DoFHandler>
static typename DoFHandler::active_cell_iterator find_2d_boundary_cell(DoFHandler &dof_handler,
                                                                       const unsigned int face_index)
{
  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    if (!cell->at_boundary(face_index)) continue;

    const auto interior_face = GeometryInfo<2>::opposite_face[face_index];
    if (cell->at_boundary(interior_face)) continue;
    const auto first_interior = cell->neighbor(interior_face);
    if (first_interior->at_boundary(interior_face)) continue;
    return cell;
  }
  return typename DoFHandler::active_cell_iterator(dof_handler.end());
}

template <typename DoFHandler>
static typename DoFHandler::active_cell_iterator find_2d_non_corner_boundary_cell(DoFHandler &dof_handler,
                                                                                  const unsigned int face_index)
{
  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    if (!cell->at_boundary(face_index)) continue;

    bool on_other_boundary = false;
    for (const auto other_face_index : cell->face_indices()) {
      if (other_face_index == face_index) continue;
      on_other_boundary = on_other_boundary || cell->at_boundary(other_face_index);
    }
    if (on_other_boundary) continue;

    const auto interior_face = GeometryInfo<2>::opposite_face[face_index];
    if (cell->at_boundary(interior_face)) continue;
    const auto first_interior = cell->neighbor(interior_face);
    if (first_interior->at_boundary(interior_face)) continue;
    return cell;
  }
  return typename DoFHandler::active_cell_iterator(dof_handler.end());
}

TEST_CASE("KT 1D FV readouts use EoM potential reconstruction and write a time series", "[FV][KT][EoM][output]")
{
  using Model = DiFfRG::Testing::ModelBurgersKT<1>;
  // Serial on purpose: this tests serial linear algebra, and a plain RectangularMesh is partitioned in an MPI build.
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_eom_potential_output_json();
  DiFfRG::Testing::PhysicalParameters prm;
  prm.initial_x0[0] = -1.0;
  prm.initial_x1[0] = 2.5;

  Model model(prm);
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  DiFfRG::FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  const std::string output_name = "kt_eom";
  auto data_out_path =
      DiFfRG::OutputPath::temporary(DiFfRG::TemporaryRetention::remove_on_destruction, output_name, "output");
  const auto output_dir = data_out_path.root();
  {
    DiFfRG::OutputSession<Assembler> data_out(data_out_path, json);
    const auto write_frame = [&](const double time) {
      data_out.write_frame(time, [&](auto &frame) {
        assembler.attach_data_output(frame, state.spatial_data(), VectorType());
        auto auxiliary = frame.fields("auxiliary");
        auxiliary.attach(discretization.get_dof_handler(), state.spatial_data(), "auxiliary_u");
      });
    };
    write_frame(0.25);
    write_frame(0.5);
  }

  const auto main_pvd = output_dir / (output_name + ".pvd");
  const auto potential_pvd = output_dir / (output_name + "_potential.pvd");
  const auto potential_vtu_0 = output_dir / "output" / (output_name + "_potential_000000.vtu");
  const auto potential_vtu_1 = output_dir / "output" / (output_name + "_potential_000001.vtu");
  const auto eom_potential_pvd = output_dir / (output_name + "_eom_potential.pvd");
  const auto eom_potential_vtu_0 = output_dir / "output" / (output_name + "_eom_potential_000000.vtu");
  const auto eom_potential_vtu_1 = output_dir / "output" / (output_name + "_eom_potential_000001.vtu");
  const auto auxiliary_pvd = output_dir / (output_name + "_auxiliary.pvd");
  const auto auxiliary_vtu_0 = output_dir / "output" / (output_name + "_auxiliary_000000.vtu");
  const auto auxiliary_vtu_1 = output_dir / "output" / (output_name + "_auxiliary_000001.vtu");

  REQUIRE(std::filesystem::exists(main_pvd));
  REQUIRE(std::filesystem::exists(potential_pvd));
  REQUIRE(std::filesystem::exists(potential_vtu_0));
  REQUIRE(std::filesystem::exists(potential_vtu_1));
  REQUIRE(std::filesystem::exists(eom_potential_pvd));
  REQUIRE(std::filesystem::exists(eom_potential_vtu_0));
  REQUIRE(std::filesystem::exists(eom_potential_vtu_1));
  REQUIRE(std::filesystem::exists(auxiliary_pvd));
  REQUIRE(std::filesystem::exists(auxiliary_vtu_0));
  REQUIRE(std::filesystem::exists(auxiliary_vtu_1));

  const auto pvd_contents = read_text_file(potential_pvd);
  CHECK(pvd_contents.find("timestep=\"0.25\"") != std::string::npos);
  CHECK(pvd_contents.find("timestep=\"0.5\"") != std::string::npos);
  CHECK(pvd_contents.find("kt_eom_potential_000000.vtu") != std::string::npos);
  CHECK(pvd_contents.find("kt_eom_potential_000001.vtu") != std::string::npos);
  const auto potential_vtu_0_contents = read_text_file(potential_vtu_0);
  const auto potential_vtu_1_contents = read_text_file(potential_vtu_1);
  CHECK(potential_vtu_0_contents.find("Name=\"potential\"") != std::string::npos);
  CHECK(potential_vtu_1_contents.find("Name=\"potential\"") != std::string::npos);
  CHECK(potential_vtu_0_contents.find("potential_hessian") == std::string::npos);
  CHECK(potential_vtu_1_contents.find("potential_hessian") == std::string::npos);

  const auto eom_pvd_contents = read_text_file(eom_potential_pvd);
  CHECK(eom_pvd_contents.find("timestep=\"0.25\"") != std::string::npos);
  CHECK(eom_pvd_contents.find("timestep=\"0.5\"") != std::string::npos);
  CHECK(eom_pvd_contents.find("kt_eom_eom_potential_000000.vtu") != std::string::npos);
  CHECK(eom_pvd_contents.find("kt_eom_eom_potential_000001.vtu") != std::string::npos);
  CHECK(read_text_file(eom_potential_vtu_0).find("Name=\"eom_potential\"") != std::string::npos);
  CHECK(read_text_file(eom_potential_vtu_1).find("Name=\"eom_potential\"") != std::string::npos);
  CHECK(read_text_file(auxiliary_vtu_0).find("Name=\"auxiliary_u\"") != std::string::npos);
  CHECK(read_text_file(auxiliary_vtu_1).find("Name=\"auxiliary_u\"") != std::string::npos);
}

TEST_CASE("KT HDF5-only output keeps all field series in one file", "[FV][KT][EoM][output][hdf5]")
{
  using Model = DiFfRG::Testing::ModelBurgersKT<1>;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_eom_potential_output_json();
  json.set_bool("/output/vtk", false);
  json.set_bool("/output/hdf5", true);

  DiFfRG::Testing::PhysicalParameters prm;
  prm.initial_x0[0] = -1.0;
  prm.initial_x1[0] = 2.5;

  Model model(prm);
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  DiFfRG::FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  const std::string output_name = "kt_eom_hdf5";
  auto data_out_path =
      DiFfRG::OutputPath::temporary(DiFfRG::TemporaryRetention::remove_on_destruction, output_name, "output");
  const auto output_dir = data_out_path.root();
  {
    DiFfRG::OutputSession<Assembler> data_out(data_out_path, json);
    const auto write_frame = [&](const double time) {
      data_out.write_frame(time, [&](auto &frame) {
        assembler.attach_data_output(frame, state.spatial_data(), VectorType());
        auto auxiliary = frame.fields("auxiliary");
        auxiliary.attach(discretization.get_dof_handler(), state.spatial_data(), "auxiliary_u");
      });
    };
    write_frame(0.25);
    write_frame(0.5);
  }

  const auto hdf5_path = output_dir / (output_name + ".h5");
  REQUIRE(std::filesystem::exists(hdf5_path));
  CHECK_FALSE(std::filesystem::exists(output_dir / (output_name + ".pvd")));
  CHECK_FALSE(std::filesystem::exists(output_dir / (output_name + "_potential.h5")));
  CHECK_FALSE(std::filesystem::exists(output_dir / (output_name + "_eom_potential.h5")));
  CHECK_FALSE(std::filesystem::exists(output_dir / (output_name + "_auxiliary.h5")));

  auto file = DiFfRG::hdf5::File::open(hdf5_path.string(), DiFfRG::hdf5::Access::ReadOnly);
  auto root = file.root();
  for (const auto &[group_name, field_name, series_name] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
           {"FE", "u", output_name},
           {"potential", "potential", output_name + "_potential"},
           {"eom_potential", "eom_potential", output_name + "_eom_potential"},
           {"auxiliary", "auxiliary_u", output_name + "_auxiliary"}}) {
    auto series = root.open_group(group_name);
    for (unsigned int frame = 0; frame < 2; ++frame) {
      auto entry = series.open_group(Utilities::int_to_string(frame, 6));
      CHECK(entry.read_attribute<double>("time") == Catch::Approx(frame == 0 ? 0.25 : 0.5));
      CHECK(entry.read_attribute<std::string>("output_name") == series_name);
      CHECK_NOTHROW(entry.open_dataset("nodes"));
      CHECK_NOTHROW(entry.open_dataset(field_name));
    }
  }
}

TEST_CASE("KT reconstruction cache recomputes cell stencil values per solution", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_test_json();
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType first(discretization.get_dof_handler().n_dofs());
  VectorType second(first.size());
  for (unsigned int i = 0; i < first.size(); ++i) {
    first[i] = 1.0 + 0.1 * static_cast<double>(i);
    second[i] = 2.0 + 0.3 * static_cast<double>(i);
  }

  auto cell = discretization.get_dof_handler().begin_active();
  ++cell;
  REQUIRE(cell != discretization.get_dof_handler().end());

  typename Assembler::SolutionReconstructionCache first_cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(first, first_cache);
  typename Assembler::SolutionReconstructionCache second_cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(second, second_cache);

  const auto cell_index = cell->active_cell_index();
  const auto &first_stencil = first_cache.cell_stencils[cell_index];
  const auto &second_stencil = second_cache.cell_stencils[cell_index];

  CHECK(first_stencil.cell.x == second_stencil.cell.x);
  CHECK(first_stencil.cell.dof_indices == second_stencil.cell.dof_indices);
  CHECK(first_stencil.cell.u[0] == Catch::Approx(first[first_stencil.cell.dof_indices[0]]));
  CHECK(second_stencil.cell.u[0] == Catch::Approx(second[second_stencil.cell.dof_indices[0]]));
  CHECK(first_stencil.cell.u[0] != Catch::Approx(second_stencil.cell.u[0]));

  for (const auto face_index : cell->face_indices()) {
    CHECK(first_stencil.neighbors.x[face_index] == second_stencil.neighbors.x[face_index]);
    CHECK(first_stencil.neighbors.dof_indices[face_index] == second_stencil.neighbors.dof_indices[face_index]);
    const auto neighbor_dof = first_stencil.neighbors.dof_indices[face_index][0];
    CHECK(first_stencil.neighbors.u[face_index][0] == Catch::Approx(first[neighbor_dof]));
    CHECK(second_stencil.neighbors.u[face_index][0] == Catch::Approx(second[neighbor_dof]));
    CHECK(first_stencil.neighbors.u[face_index][0] != Catch::Approx(second_stencil.neighbors.u[face_index][0]));
  }

  const auto face_index = 1U;
  REQUIRE(first_cache.face_reconstruction_valid[cell_index][face_index]);
  REQUIRE(second_cache.face_reconstruction_valid[cell_index][face_index]);
  const auto &first_reconstruction = assembler.get_cached_face_reconstruction(first_cache, cell, face_index);
  const auto &second_reconstruction = assembler.get_cached_face_reconstruction(second_cache, cell, face_index);
  CHECK(first_reconstruction.u_minus[0] != Catch::Approx(second_reconstruction.u_minus[0]));
  CHECK(first_reconstruction.u_plus[0] != Catch::Approx(second_reconstruction.u_plus[0]));
}

TEST_CASE("KT solution reconstruction cache stores reversed interior face orientations", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_test_json();
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 1.0 + 0.13 * static_cast<double>(i) + 0.04 * static_cast<double>(i * i);

  auto cell = discretization.get_dof_handler().begin_active();
  ++cell;
  REQUIRE(cell != discretization.get_dof_handler().end());

  const auto face_index = 1U;
  REQUIRE(!cell->at_boundary(face_index));
  const auto neighbor = cell->neighbor(face_index);
  const auto neighbor_face_index = assembler.find_neighbor_face(cell, neighbor);

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);
  REQUIRE(cache.face_reconstruction_valid[cell->active_cell_index()][face_index]);
  REQUIRE(cache.face_reconstruction_valid[neighbor->active_cell_index()][neighbor_face_index]);

  const auto &cell_state = assembler.get_cached_face_reconstruction(cache, cell, face_index);
  const auto &neighbor_state = assembler.get_cached_face_reconstruction(cache, neighbor, neighbor_face_index);
  check_face_reconstruction_state_reversed_1d(cell_state, neighbor_state);
}

TEST_CASE("KT solution reconstruction cache matches direct interior reconstruction", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_test_json();
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 0.7 + 0.21 * static_cast<double>(i) - 0.02 * static_cast<double>(i * i);

  auto cell = discretization.get_dof_handler().begin_active();
  ++cell;
  REQUIRE(cell != discretization.get_dof_handler().end());

  const auto face_index = 1U;
  REQUIRE(!cell->at_boundary(face_index));
  const auto neighbor = cell->neighbor(face_index);
  const auto x_q = cell->face(face_index)->center();

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);
  const auto &cached = assembler.get_cached_face_reconstruction(cache, cell, face_index);

  typename Assembler::CellStencilData cell_stencil;
  typename Assembler::CellStencilData neighbor_stencil;
  assembler.fill_cell_stencil(cell, solution, cell_stencil);
  assembler.fill_cell_stencil(neighbor, solution, neighbor_stencil);
  const auto expected = KT::internal::compute_interior_face_reconstruction_state<typename Assembler::Reconstructor>(
      cell_stencil, neighbor_stencil, x_q);

  check_face_reconstruction_state_1d(cached, expected);
}

TEST_CASE("KT solution reconstruction cache matches direct boundary reconstruction", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_test_json();
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 0.4 + 0.17 * static_cast<double>(i) + 0.03 * static_cast<double>(i * i);

  const auto cell = discretization.get_dof_handler().begin_active();
  const auto face_index = 0U;
  REQUIRE(cell->at_boundary(face_index));
  const auto x_q = cell->face(face_index)->center();

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);
  const auto &cached = assembler.get_cached_face_reconstruction(cache, cell, face_index);

  const auto boundary_stencil =
      assembler.template build_boundary_stencil_from_cache<NumberType>(cell, face_index, solution);
  const auto expected = KT::internal::compute_boundary_face_reconstruction_state<typename Assembler::Reconstructor>(
      boundary_stencil, cache.cell_stencils[cell->active_cell_index()], x_q, model);

  check_face_reconstruction_state_1d(cached, expected);
}

TEST_CASE("KT boundary stencil cache recomputes values per solution", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;
  using BoundaryIndex = DiFfRG::FV::KurganovTadmor::internal::BoundaryStencilIndex<1>;

  ensure_logger();

  auto json = make_fv_test_json();
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType first(discretization.get_dof_handler().n_dofs());
  VectorType second(first.size());
  for (unsigned int i = 0; i < first.size(); ++i) {
    first[i] = 1.0 + 0.2 * static_cast<double>(i);
    second[i] = 3.0 + 0.4 * static_cast<double>(i);
  }

  const auto cell = discretization.get_dof_handler().begin_active();
  const auto first_boundary = assembler.template build_boundary_stencil_from_cache<NumberType>(cell, 0, first);
  const auto second_boundary = assembler.template build_boundary_stencil_from_cache<NumberType>(cell, 0, second);

  for (const auto stencil_index :
       {BoundaryIndex::physical_cell, BoundaryIndex::upper_inner, BoundaryIndex::upper_outer}) {
    CHECK(first_boundary.x[stencil_index] == second_boundary.x[stencil_index]);
    CHECK(first_boundary.dof_indices[stencil_index] == second_boundary.dof_indices[stencil_index]);
    const auto dof = first_boundary.dof_indices[stencil_index][0];
    CHECK(first_boundary.u[stencil_index][0] == Catch::Approx(first[dof]));
    CHECK(second_boundary.u[stencil_index][0] == Catch::Approx(second[dof]));
    CHECK(first_boundary.u[stencil_index][0] != Catch::Approx(second_boundary.u[stencil_index][0]));
  }
}

TEST_CASE("KT 2D topology cache stores four face neighbours for an interior cell", "[FV][KT][cache][2d]")
{
  using Model = DiFfRG::Testing::ModelBurgers2DKT;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<2>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;

  ensure_logger();

  auto json = make_kt_boundary_json();
  DiFfRG::Testing::PhysicalParameters prm;
  prm.initial_x0[0] = 1.0;
  prm.initial_x1[0] = 0.25;
  prm.initial_x2[0] = 0.5;

  Model model(prm);
  DiFfRG::RectangularMeshSerial<2> mesh{DiFfRG::Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  auto cell = find_2d_interior_cell(discretization.get_dof_handler());
  REQUIRE(cell != discretization.get_dof_handler().end());

  typename Assembler::CellStencilData stencil;
  typename Discretization::VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 1.0 + 0.2 * static_cast<double>(i);
  assembler.fill_cell_stencil(cell, solution, stencil);

  CHECK(stencil.cell.x == cell->center());
  for (const auto face_index : cell->face_indices()) {
    CAPTURE(face_index);
    REQUIRE(!cell->at_boundary(face_index));
    CHECK(stencil.face_centers[face_index] == cell->face(face_index)->center());
    CHECK(stencil.neighbors.x[face_index] == cell->neighbor(face_index)->center());
    CHECK(stencil.neighbors.dof_indices[face_index][0] != dealii::numbers::invalid_dof_index);
    CHECK(stencil.neighbors.u[face_index][0] == Catch::Approx(solution[stencil.neighbors.dof_indices[face_index][0]]));
  }
}

TEST_CASE("KT 2D boundary stencil cache stores two interior cells behind boundary faces", "[FV][KT][cache][2d]")
{
  using Model = DiFfRG::Testing::ModelBurgers2DKT;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<2>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using namespace DiFfRG::def::BoundaryStencilIndex;

  ensure_logger();

  auto json = make_kt_boundary_json();
  DiFfRG::Testing::PhysicalParameters prm;
  prm.initial_x0[0] = 1.0;
  prm.initial_x1[0] = 0.25;
  prm.initial_x2[0] = 0.5;

  Model model(prm);
  DiFfRG::RectangularMeshSerial<2> mesh{DiFfRG::Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  typename Discretization::VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 1.0 + 0.2 * static_cast<double>(i);

  for (const auto face_index : {0U, 1U, 2U, 3U}) {
    CAPTURE(face_index);
    auto cell = find_2d_boundary_cell(discretization.get_dof_handler(), face_index);
    REQUIRE(cell != discretization.get_dof_handler().end());

    const auto interior_face = GeometryInfo<2>::opposite_face[face_index];
    const auto first_interior = cell->neighbor(interior_face);
    const auto second_interior = first_interior->neighbor(interior_face);

    const auto boundary_stencil =
        assembler.template build_boundary_stencil_from_cache<NumberType>(cell, face_index, solution);

    CHECK(boundary_stencil.cell_face == face_index);
    CHECK(boundary_stencil.lower_boundary == (face_index % 2 == 0));
    CHECK(boundary_stencil.x[physical_cell] == cell->center());
    CHECK(boundary_stencil.dof_indices[physical_cell][0] != dealii::numbers::invalid_dof_index);
    CHECK(boundary_stencil.u[physical_cell][0] ==
          Catch::Approx(solution[boundary_stencil.dof_indices[physical_cell][0]]));

    const auto first_index = boundary_stencil.lower_boundary ? upper_inner : lower_inner;
    const auto second_index = boundary_stencil.lower_boundary ? upper_outer : lower_outer;
    CHECK(boundary_stencil.x[first_index] == first_interior->center());
    CHECK(boundary_stencil.x[second_index] == second_interior->center());
    CHECK(boundary_stencil.dof_indices[first_index][0] != dealii::numbers::invalid_dof_index);
    CHECK(boundary_stencil.dof_indices[second_index][0] != dealii::numbers::invalid_dof_index);
    CHECK(boundary_stencil.u[first_index][0] == Catch::Approx(solution[boundary_stencil.dof_indices[first_index][0]]));
    CHECK(boundary_stencil.u[second_index][0] ==
          Catch::Approx(solution[boundary_stencil.dof_indices[second_index][0]]));
  }
}

TEST_CASE("KT 2D solution reconstruction cache recomputes values per solution", "[FV][KT][cache][2d]")
{
  using Model = DiFfRG::Testing::ModelBurgers2DKT;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<2>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_kt_boundary_json();
  DiFfRG::Testing::PhysicalParameters prm;
  prm.initial_x0[0] = 1.0;
  prm.initial_x1[0] = 0.25;
  prm.initial_x2[0] = 0.5;

  Model model(prm);
  DiFfRG::RectangularMeshSerial<2> mesh{DiFfRG::Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType first(discretization.get_dof_handler().n_dofs());
  VectorType second(first.size());
  for (unsigned int i = 0; i < first.size(); ++i) {
    first[i] = 1.0 + 0.1 * static_cast<double>(i);
    second[i] = 2.0 + 0.3 * static_cast<double>(i);
  }

  auto cell = find_2d_interior_cell(discretization.get_dof_handler());
  REQUIRE(cell != discretization.get_dof_handler().end());

  typename Assembler::SolutionReconstructionCache first_cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(first, first_cache);
  typename Assembler::SolutionReconstructionCache second_cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(second, second_cache);

  const auto cell_index = cell->active_cell_index();
  const auto &first_stencil = first_cache.cell_stencils[cell_index];
  const auto &second_stencil = second_cache.cell_stencils[cell_index];

  CHECK(first_stencil.cell.x == second_stencil.cell.x);
  CHECK(first_stencil.cell.dof_indices == second_stencil.cell.dof_indices);
  CHECK(first_stencil.cell.u[0] == Catch::Approx(first[first_stencil.cell.dof_indices[0]]));
  CHECK(second_stencil.cell.u[0] == Catch::Approx(second[second_stencil.cell.dof_indices[0]]));
  CHECK(first_stencil.cell.u[0] != Catch::Approx(second_stencil.cell.u[0]));

  for (const auto face_index : cell->face_indices()) {
    REQUIRE(first_cache.face_reconstruction_valid[cell_index][face_index]);
    REQUIRE(second_cache.face_reconstruction_valid[cell_index][face_index]);
    const auto &first_reconstruction = assembler.get_cached_face_reconstruction(first_cache, cell, face_index);
    const auto &second_reconstruction = assembler.get_cached_face_reconstruction(second_cache, cell, face_index);
    CHECK(first_reconstruction.u_minus[0] != Catch::Approx(second_reconstruction.u_minus[0]));
  }
}

TEST_CASE("KT 2D solution reconstruction cache matches direct interior reconstruction", "[FV][KT][cache][2d]")
{
  using Model = DiFfRG::Testing::ModelBurgers2DKT;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<2>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_kt_boundary_json();
  DiFfRG::Testing::PhysicalParameters prm;
  prm.initial_x0[0] = 1.0;
  prm.initial_x1[0] = 0.25;
  prm.initial_x2[0] = 0.5;

  Model model(prm);
  DiFfRG::RectangularMeshSerial<2> mesh{DiFfRG::Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 0.7 + 0.21 * static_cast<double>(i) - 0.02 * static_cast<double>(i * i);

  auto cell = find_2d_interior_cell(discretization.get_dof_handler());
  REQUIRE(cell != discretization.get_dof_handler().end());

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);

  for (const auto face_index : {1U, 3U}) {
    CAPTURE(face_index);
    REQUIRE(!cell->at_boundary(face_index));
    const auto neighbor = cell->neighbor(face_index);
    const auto x_q = cell->face(face_index)->center();
    const auto &cached = assembler.get_cached_face_reconstruction(cache, cell, face_index);

    typename Assembler::CellStencilData cell_stencil;
    typename Assembler::CellStencilData neighbor_stencil;
    assembler.fill_cell_stencil(cell, solution, cell_stencil);
    assembler.fill_cell_stencil(neighbor, solution, neighbor_stencil);
    const auto expected = KT::internal::compute_interior_face_reconstruction_state<typename Assembler::Reconstructor>(
        cell_stencil, neighbor_stencil, x_q);

    check_face_reconstruction_state<2>(cached, expected);
  }
}

TEST_CASE("KT 2D boundary reconstruction cache reconstructs affine ghost side", "[FV][KT][cache][2d]")
{
  using Model = DiFfRG::Testing::ModelBurgers2DKT;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<2>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_kt_boundary_json();
  DiFfRG::Testing::PhysicalParameters prm;
  prm.initial_x0[0] = 1.0;
  prm.initial_x1[0] = 0.25;
  prm.initial_x2[0] = 0.5;

  Model model(prm);
  DiFfRG::RectangularMeshSerial<2> mesh{DiFfRG::Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  const auto &support_points = discretization.get_support_points();
  REQUIRE(support_points.size() == solution.size());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = model.solution(support_points[i])[0];

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);

  const auto check_boundary_face = [&](const auto &cell, const unsigned int face_index) {
    CAPTURE(face_index, cell->center());
    REQUIRE(cell != discretization.get_dof_handler().end());
    REQUIRE(cell->at_boundary(face_index));

    const auto x_q = cell->face(face_index)->center();
    const auto &cached = assembler.get_cached_face_reconstruction(cache, cell, face_index);
    const double expected_value = model.solution(x_q)[0];
    constexpr double tolerance = 1.0e-12;

    CHECK(cached.u_minus[0] == Catch::Approx(expected_value).margin(tolerance));
    CHECK(cached.u_plus[0] == Catch::Approx(expected_value).margin(tolerance));
    CHECK(cached.face_grad_minus[0][0] == Catch::Approx(prm.initial_x1[0]).margin(tolerance));
    CHECK(cached.face_grad_minus[0][1] == Catch::Approx(prm.initial_x2[0]).margin(tolerance));
    CHECK(cached.face_grad_plus[0][0] == Catch::Approx(prm.initial_x1[0]).margin(tolerance));
    CHECK(cached.face_grad_plus[0][1] == Catch::Approx(prm.initial_x2[0]).margin(tolerance));
  };

  auto side_cell = find_2d_non_corner_boundary_cell(discretization.get_dof_handler(), 0U);
  check_boundary_face(side_cell, 0U);

  auto corner_cell = find_2d_boundary_cell(discretization.get_dof_handler(), 2U);
  REQUIRE(corner_cell != discretization.get_dof_handler().end());
  REQUIRE((corner_cell->at_boundary(0U) || corner_cell->at_boundary(1U)));
  check_boundary_face(corner_cell, 2U);
}

TEST_CASE("KT 2D boundary reconstruction cache derives tangential ghost side from named slots", "[FV][KT][cache][2d]")
{
  using Model = NamedSlotAffineBoundary2DModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<2>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_kt_boundary_json();
  Model model;
  DiFfRG::RectangularMeshSerial<2> mesh{DiFfRG::Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  const auto &support_points = discretization.get_support_points();
  REQUIRE(support_points.size() == solution.size());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = model.solution(support_points[i])[0];

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);

  for (const auto face_index : {0U, 1U, 2U, 3U}) {
    CAPTURE(face_index);
    const auto cell = find_2d_non_corner_boundary_cell(discretization.get_dof_handler(), face_index);
    REQUIRE(cell != discretization.get_dof_handler().end());
    REQUIRE(cell->at_boundary(face_index));

    const auto x_q = cell->face(face_index)->center();
    const auto &cached = assembler.get_cached_face_reconstruction(cache, cell, face_index);
    constexpr double tolerance = 1.0e-12;

    CHECK(cached.u_minus[0] == Catch::Approx(model.solution(x_q)[0]).margin(tolerance));
    CHECK(cached.u_plus[0] == Catch::Approx(model.solution(x_q)[0]).margin(tolerance));
    CHECK(cached.face_grad_minus[0][0] == Catch::Approx(Model::x_slope).margin(tolerance));
    CHECK(cached.face_grad_minus[0][1] == Catch::Approx(Model::y_slope).margin(tolerance));
    CHECK(cached.face_grad_plus[0][0] == Catch::Approx(Model::x_slope).margin(tolerance));
    CHECK(cached.face_grad_plus[0][1] == Catch::Approx(Model::y_slope).margin(tolerance));
  }
}

TEST_CASE("KT 2D boundary reconstruction cache uses model-owned tangential ghost neighbours", "[FV][KT][cache][2d]")
{
  using Model = NonAffineTangentialGhostBoundary2DModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<2>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_kt_boundary_json();
  Model model;
  DiFfRG::RectangularMeshSerial<2> mesh{DiFfRG::Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  const auto &support_points = discretization.get_support_points();
  REQUIRE(support_points.size() == solution.size());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = model.solution(support_points[i])[0];

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);

  for (const auto face_index : {0U, 1U, 2U, 3U}) {
    CAPTURE(face_index);
    const auto cell = find_2d_non_corner_boundary_cell(discretization.get_dof_handler(), face_index);
    REQUIRE(cell != discretization.get_dof_handler().end());
    REQUIRE(cell->at_boundary(face_index));

    const auto &cached = assembler.get_cached_face_reconstruction(cache, cell, face_index);
    const unsigned int normal_axis = face_index / 2;
    const unsigned int tangential_axis = 1U - normal_axis;
    const double base_tangential_slope = tangential_axis == 0 ? Model::x_slope : Model::y_slope;
    constexpr double tolerance = 1.0e-12;

    CHECK(cached.face_grad_minus[0][tangential_axis] == Catch::Approx(base_tangential_slope).margin(tolerance));
    CHECK(cached.face_grad_plus[0][tangential_axis] ==
          Catch::Approx(base_tangential_slope + Model::tangential_shift_slope).margin(tolerance));
  }
}

TEST_CASE("KT origin-odd boundary reconstruction remains curl-free", "[FV][KT][cache][boundary][2d]")
{
  using Model = OriginOddCurlFreeBoundary2DModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<2>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_kt_boundary_json();
  json.set_string("/discretization/grid/x_grid", "-0.125:0.25:0.875");
  json.set_string("/discretization/grid/y_grid", "-0.125:0.25:0.875");

  Model model;
  const DiFfRG::Config::ConfigurationMesh<2> mesh_config(json);
  DiFfRG::RectangularMeshSerial<2> mesh(mesh_config);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  DiFfRG::FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &solution = state.spatial_data();

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);

  const auto check_gradient = [](const auto &gradient) {
    constexpr double tolerance = 1.0e-12;
    CHECK(gradient[0][0] == Catch::Approx(Model::x_slope).margin(tolerance));
    CHECK(gradient[1][1] == Catch::Approx(Model::y_slope).margin(tolerance));
    CHECK(gradient[0][1] == Catch::Approx(0.0).margin(tolerance));
    CHECK(gradient[1][0] == Catch::Approx(0.0).margin(tolerance));
    const double curl = gradient[1][0] - gradient[0][1];
    CHECK(curl == Catch::Approx(0.0).margin(tolerance));
  };

  for (const auto face_index : {0U, 2U}) {
    CAPTURE(face_index);
    const auto cell = find_2d_non_corner_boundary_cell(discretization.get_dof_handler(), face_index);
    REQUIRE(cell != discretization.get_dof_handler().end());

    const auto x_q = cell->face(face_index)->center();
    const auto expected = model.solution(x_q);
    const auto &reconstruction = assembler.get_cached_face_reconstruction(cache, cell, face_index);

    for (std::size_t component = 0; component < expected.size(); ++component) {
      CHECK(reconstruction.u_minus[component] == Catch::Approx(expected[component]).margin(1.0e-12));
      CHECK(reconstruction.u_plus[component] == Catch::Approx(expected[component]).margin(1.0e-12));
    }

    check_gradient(reconstruction.face_grad_minus);
    check_gradient(reconstruction.face_grad_plus);
    check_gradient(reconstruction.diffusion_grad_minus);
    check_gradient(reconstruction.diffusion_grad_plus);
    check_gradient(reconstruction.diffusion_grad);
  }
}

TEST_CASE("KT 2D solution reconstruction cache stores reversed interior face orientations", "[FV][KT][cache][2d]")
{
  using Model = DiFfRG::Testing::ModelBurgers2DKT;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<2>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_kt_boundary_json();
  DiFfRG::Testing::PhysicalParameters prm;
  prm.initial_x0[0] = 1.0;
  prm.initial_x1[0] = 0.25;
  prm.initial_x2[0] = 0.5;

  Model model(prm);
  DiFfRG::RectangularMeshSerial<2> mesh{DiFfRG::Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 1.0 + 0.13 * static_cast<double>(i) + 0.04 * static_cast<double>(i * i);

  auto cell = find_2d_interior_cell(discretization.get_dof_handler());
  REQUIRE(cell != discretization.get_dof_handler().end());

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);

  for (const auto face_index : {1U, 3U}) {
    CAPTURE(face_index);
    REQUIRE(!cell->at_boundary(face_index));
    const auto neighbor = cell->neighbor(face_index);
    const auto neighbor_face_index = assembler.find_neighbor_face(cell, neighbor);
    const auto &cell_state = assembler.get_cached_face_reconstruction(cache, cell, face_index);
    const auto &neighbor_state = assembler.get_cached_face_reconstruction(cache, neighbor, neighbor_face_index);
    check_face_reconstruction_state_reversed<2>(cell_state, neighbor_state);
  }
}

TEST_CASE("KT face hook exposes cached reconstruction before residual flux", "[FV][KT][hook]")
{
  using Model = FaceHookProbeModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_test_json();
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  DiFfRG::FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &solution = state.spatial_data();

  typename Assembler::SolutionReconstructionCache cache;
  assembler.template rebuild_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution, cache);
  const auto context = assembler.make_assembly_context_view(cache);
  STATIC_REQUIRE(KT::HasAssemblyContextView<decltype(context)>);
  STATIC_REQUIRE(!CanMakeAssemblyContextViewFromTemporaryCache<Assembler>);

  VectorType residual(solution.size());
  VectorType solution_dot(solution.size());
  residual = 0.;
  solution_dot = 0.;

  assembler.residual(residual, solution, 1., solution_dot, 0.);

  CHECK(model.hook_calls == 1);
  CHECK(model.hook_stage == KT::AssemblyStage::residual);
  CHECK(model.hook_face_count == solution.size() + 1);
  CHECK(model.hook_cell_count == solution.size());
  CHECK(model.saw_boundary_face);
  CHECK(model.saw_interior_face);
  CHECK(std::isfinite(model.hook_min_u_plus));
  CHECK(model.hook_parallel_min_u_plus == Catch::Approx(model.hook_min_u_plus).margin(1e-14));
  CHECK(std::isfinite(model.hook_min_cell_u));
  CHECK(std::isfinite(model.last_u_minus));
  CHECK(std::isfinite(model.last_u_plus));
  CHECK(std::isfinite(model.last_grad_minus));
  CHECK(std::isfinite(model.last_grad_plus));
  CHECK(std::isfinite(model.last_cell_u));
  CHECK(std::isfinite(model.last_cell_point));
  CHECK(model.flux_calls > 0);
  CHECK(model.flux_saw_hook_state);
  CHECK(model.kt_flux_gradients_match);
  CHECK(model.kt_flux_saw_boundary_gradient);
  CHECK(model.kt_flux_saw_interior_gradient);
}

TEST_CASE("KT diffusion flux receives reconstructed third derivatives", "[FV][KT]")
{
  using Model = ThirdDerivativeFluxProbeModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_test_json();
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  DiFfRG::FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &solution = state.spatial_data();

  VectorType residual(solution.size());
  VectorType solution_dot(solution.size());
  residual = 0.;
  solution_dot = 0.;

  assembler.residual(residual, solution, 1., solution_dot, 0.);

  CHECK(model.saw_third_derivative);
  CHECK(model.saw_exact_cubic_third_derivative);
  CHECK(std::isfinite(model.min_third_derivative));
  CHECK(std::isfinite(model.max_third_derivative));
}

TEST_CASE("u_plus u_minus compoutation", "[FV][KT]")
{
  const int dim = 1;
  const uint n_components = 2;
  using GradComponentType = dealii::Tensor<1, dim, NumberType>;

  const DiFfRG::FV::KurganovTadmor::internal::GradientType<dim, NumberType, n_components> u_grad(
      {GradComponentType({-0.5}), GradComponentType({0.7})});
  const Point<dim> x_center(1.0);
  const Point<dim> x_q(2.0);
  const std::array<NumberType, n_components> u_val = {1.0, 2.0};
  const std::array<NumberType, n_components> u_minus_reference({0.5, 2.7});

  const std::array<NumberType, n_components> u_reconstructed = reconstruct_u(u_val, x_center, x_q, u_grad);
  CHECK(u_minus_reference[0] == Catch::Approx(u_reconstructed[0]));
  CHECK(u_minus_reference[1] == Catch::Approx(u_reconstructed[1]));
}

TEST_CASE("reconstruct_u_derivative 1D single component", "[FV][KT]")
{
  // 1D uniform grid: center at x=0, left neighbor at x=-1, right neighbor at x=1
  // u_center=2, u_left=1, u_right=4
  // slopes: du_left = (1-2)/(-1) = 1, du_right = (4-2)/1 = 2
  // minmod(1, 2) = 1 (picks du_left since |1|<|2|), gradient = 1
  // Reconstruct at x_q = 0.5: u_recon = 2 + 1*0.5 = 2.5
  constexpr int dim = 1;
  constexpr size_t n_components = 1;
  using AD = autodiff::Real<1, NumberType>;
  using Reconstructor = DiFfRG::def::TVDReconstructor<dim, DiFfRG::def::MinModLimiter, NumberType>;

  const Point<dim> center(0.0);
  const Point<dim> x_q(0.5);
  const std::array<Point<dim>, 2> x_n = {Point<dim>(-1.0), Point<dim>(1.0)};

  SECTION("derivative w.r.t. center value")
  {
    // minmod picks du_left = (u_left-u_c)/(-1).
    // d(du_left)/d(u_c) = -1/(-1) = 1, so d(grad)/d(u_c) = 1
    // d(u_recon)/d(u_c) = 1 + 1 * 0.5 = 1.5
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_c[0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_c[0]);
    CHECK(result[0] == Catch::Approx(1.5));
  }

  SECTION("derivative w.r.t. left neighbor")
  {
    // d(du_left)/d(u_left) = 1/(-1) = -1, minmod picks du_left
    // d(grad)/d(u_left) = -1
    // d(u_recon)/d(u_left) = 0 + (-1) * 0.5 = -0.5
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_n[0][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[0][0]);
    CHECK(result[0] == Catch::Approx(-0.5));
  }

  SECTION("derivative w.r.t. right neighbor (inactive in minmod)")
  {
    // minmod picks du_left (not du_right), so d(grad)/d(u_right) = 0
    // d(u_recon)/d(u_right) = 0 + 0 = 0
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_n[1][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[1][0]);
    CHECK(result[0] == Catch::Approx(0.0));
  }
}

TEST_CASE("reconstruct_u_derivative at local extremum (vanishing limiter)", "[FV][KT]")
{
  // Opposite slopes => minmod returns 0, gradient derivative also 0
  // center at x=0, u=3; left at x=-1, u=4; right at x=1, u=4
  // du_left = (4-3)/(-1) = -1, du_right = (4-3)/1 = 1
  // minmod(-1, 1) = 0 => flat reconstruction
  constexpr int dim = 1;
  constexpr size_t n_components = 1;
  using AD = autodiff::Real<1, NumberType>;
  using Reconstructor = DiFfRG::def::TVDReconstructor<dim, DiFfRG::def::MinModLimiter, NumberType>;

  const Point<dim> center(0.0);
  const Point<dim> x_q(0.5);
  const std::array<Point<dim>, 2> x_n = {Point<dim>(-1.0), Point<dim>(1.0)};

  SECTION("derivative w.r.t. center is Kronecker delta only")
  {
    std::array<AD, 1> u_c = {AD(3.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(4.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_c[0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_c[0]);
    CHECK(result[0] == Catch::Approx(1.0));
  }

  SECTION("derivative w.r.t. neighbor is zero")
  {
    std::array<AD, 1> u_c = {AD(3.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(4.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_n[0][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[0][0]);
    CHECK(result[0] == Catch::Approx(0.0));
  }
}

TEST_CASE("reconstruct_u_derivative 2D single component", "[FV][KT]")
{
  // 2D: center at (0,0), u=2
  // x-neighbors: left (-1,0) u=1, right (1,0) u=4
  //   du_x_left = (1-2)/(-1)=1, du_x_right = (4-2)/1=2, minmod=1 (picks left)
  // y-neighbors: bottom (0,-1) u=0.5, top (0,1) u=5
  //   du_y_bottom = (0.5-2)/(-1)=1.5, du_y_top = (5-2)/1=3, minmod=1.5 (picks bottom)
  // gradient = (1, 1.5)
  constexpr int dim = 2;
  constexpr size_t n_components = 1;
  using AD = autodiff::Real<1, NumberType>;
  using Reconstructor = DiFfRG::def::TVDReconstructor<dim, DiFfRG::def::MinModLimiter, NumberType>;

  const Point<dim> center(0.0, 0.0);
  const Point<dim> x_q(0.5, 0.3);
  // neighbors ordered: x-left, x-right, y-left(bottom), y-right(top)
  const std::array<Point<dim>, 4> x_n = {Point<dim>(-1.0, 0.0), Point<dim>(1.0, 0.0), Point<dim>(0.0, -1.0),
                                         Point<dim>(0.0, 1.0)};

  SECTION("derivative w.r.t. center")
  {
    // d(grad_x)/d(u_c) = d(du_x_left)/d(u_c) = -1/(-1) = 1
    // d(grad_y)/d(u_c) = d(du_y_bottom)/d(u_c) = -1/(-1) = 1
    // grad_deriv = (1, 1)
    // d(u_recon)/d(u_c) = 1 + 1*0.5 + 1*0.3 = 1.8
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)},
                                            std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(5.0)}};
    seed(u_c[0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_c[0]);
    CHECK(result[0] == Catch::Approx(1.8));
  }

  SECTION("derivative w.r.t. x-left neighbor")
  {
    // d(grad_x)/d(u_xleft) = 1/(-1) = -1 (minmod picks du_x_left)
    // d(grad_y)/d(u_xleft) = 0 (y-dimension independent)
    // d(u_recon)/d(u_xleft) = 0 + (-1)*0.5 + 0*0.3 = -0.5
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)},
                                            std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(5.0)}};
    seed(u_n[0][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[0][0]);
    CHECK(result[0] == Catch::Approx(-0.5));
  }

  SECTION("derivative w.r.t. y-bottom neighbor")
  {
    // d(grad_x)/d(u_ybottom) = 0
    // d(grad_y)/d(u_ybottom) = 1/(-1) = -1 (minmod picks du_y_bottom)
    // d(u_recon)/d(u_ybottom) = 0 + 0*0.5 + (-1)*0.3 = -0.3
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)},
                                            std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(5.0)}};
    seed(u_n[2][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[2][0]);
    CHECK(result[0] == Catch::Approx(-0.3));
  }
}

TEST_CASE("Test compute_numerical_flux in 1D with one component", "[FV][KT]")
{
  constexpr int dim = 1;
  constexpr size_t n_components = 1;

  SECTION("Symmetric flux cancels diffusion term")
  {
    // u_plus == u_minus => H = F (no diffusion)
    std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({3.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({3.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({5.0})};
    std::array<NumberType, n_components> u_plus = {1.0};
    std::array<NumberType, n_components> u_minus = {1.0};

    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(3.0));
  }

  SECTION("General case")
  {
    // F_plus = 4.0, F_minus = 2.0, a = 1.0, u_plus = 3.0, u_minus = 1.0
    // H = (4+2)/2 - 1*(3-1)/2 = 3 - 1 = 2
    std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({4.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({2.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({1.0})};
    std::array<NumberType, n_components> u_plus = {3.0};
    std::array<NumberType, n_components> u_minus = {1.0};

    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(2.0));
  }

  SECTION("Zero wave speed reduces to central flux")
  {
    // a = 0 => H = (F_plus + F_minus)/2
    std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({6.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({2.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({0.0})};
    std::array<NumberType, n_components> u_plus = {5.0};
    std::array<NumberType, n_components> u_minus = {1.0};

    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(4.0));
  }
}

TEST_CASE("Test compute_numerical_flux in 2D with one component", "[FV][KT]")
{
  constexpr int dim = 2;
  constexpr size_t n_components = 1;

  // F_plus = (4, 1), F_minus = (2, 3), a = (1, 2), u_plus = 3, u_minus = 1
  // H[0] = (4+2)/2 - 1*(3-1)/2 = 3 - 1 = 2
  // H[1] = (1+3)/2 - 2*(3-1)/2 = 2 - 2 = 0
  std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({4.0, 1.0})};
  std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({2.0, 3.0})};
  std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({1.0, 2.0})};
  std::array<NumberType, n_components> u_plus = {3.0};
  std::array<NumberType, n_components> u_minus = {1.0};

  const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
  CHECK(H[0][0] == Catch::Approx(2.0));
  CHECK(H[0][1] == Catch::Approx(0.0));
}

TEST_CASE("Test compute_kt_flux_and_speeds with Burgers flux in 1D", "[FV][KT]")
{
  // TestModel: F(u) = u^2/2 + x, so dF/du = u
  constexpr int dim = 1;
  constexpr size_t n_components = 1;
  TestModel model;

  SECTION("Flux values and wave speed")
  {
    // At x=0.5, u_plus=2.0, u_minus=1.0
    // F(u_plus)  = 2^2/2 + 0.5 = 2.5
    // F(u_minus) = 1^2/2 + 0.5 = 1.0
    // dF/du(u_plus) = 2.0, dF/du(u_minus) = 1.0
    // a = max(|2|, |1|) = 2.0
    const Point<dim> x_q(0.5);
    const std::array<NumberType, n_components> u_plus = {2.0};
    const std::array<NumberType, n_components> u_minus = {1.0};

    const auto [F_plus, F_minus, a_half] = DiFfRG::Testing::compute_kt_flux_and_speeds<WaveSpeedStrategy>(
        u_plus, u_minus, x_q, test_cell_width, test_cell_width, no_extractors, no_variables, model);
    CHECK(F_plus[0][0] == Catch::Approx(2.5));
    CHECK(F_minus[0][0] == Catch::Approx(1.0));
    CHECK(a_half[0][0] == Catch::Approx(2.0));
  }

  SECTION("Symmetric states give symmetric fluxes")
  {
    // u_plus = u_minus = 3.0, x=1.0
    // F = 3^2/2 + 1 = 5.5
    // a = |3| = 3.0
    const Point<dim> x_q(1.0);
    const std::array<NumberType, n_components> u_plus = {3.0};
    const std::array<NumberType, n_components> u_minus = {3.0};

    const auto [F_plus, F_minus, a_half] = DiFfRG::Testing::compute_kt_flux_and_speeds<WaveSpeedStrategy>(
        u_plus, u_minus, x_q, test_cell_width, test_cell_width, no_extractors, no_variables, model);
    CHECK(F_plus[0][0] == Catch::Approx(5.5));
    CHECK(F_minus[0][0] == Catch::Approx(5.5));
    CHECK(a_half[0][0] == Catch::Approx(3.0));
  }

  SECTION("Negative velocities")
  {
    // u_plus = -1.0, u_minus = -3.0, x=0
    // F(u_plus)  = (-1)^2/2 + 0 = 0.5
    // F(u_minus) = (-3)^2/2 + 0 = 4.5
    // dF/du(u_plus) = -1, dF/du(u_minus) = -3
    // a = max(|-1|, |-3|) = 3.0
    const Point<dim> x_q(0.0);
    const std::array<NumberType, n_components> u_plus = {-1.0};
    const std::array<NumberType, n_components> u_minus = {-3.0};

    const auto [F_plus, F_minus, a_half] = DiFfRG::Testing::compute_kt_flux_and_speeds<WaveSpeedStrategy>(
        u_plus, u_minus, x_q, test_cell_width, test_cell_width, no_extractors, no_variables, model);
    CHECK(F_plus[0][0] == Catch::Approx(0.5));
    CHECK(F_minus[0][0] == Catch::Approx(4.5));
    CHECK(a_half[0][0] == Catch::Approx(3.0));
  }

  SECTION("Full KT numerical flux for Burgers")
  {
    // u_plus=2, u_minus=1, x=0.5
    // F_plus=2.5, F_minus=1.0, a=2.0
    // H = (2.5+1.0)/2 - 2.0*(2.0-1.0)/2 = 1.75 - 1.0 = 0.75
    const Point<dim> x_q(0.5);
    const std::array<NumberType, n_components> u_plus = {2.0};
    const std::array<NumberType, n_components> u_minus = {1.0};

    const auto [F_plus, F_minus, a_half] = DiFfRG::Testing::compute_kt_flux_and_speeds<WaveSpeedStrategy>(
        u_plus, u_minus, x_q, test_cell_width, test_cell_width, no_extractors, no_variables, model);
    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(0.75));
  }
}

// ---------------------------------------------------------------------------
// Tests for tag_cell_dofs
// ---------------------------------------------------------------------------

using KT::internal::make_tagged_neighbors;
using KT::internal::tag_cell_dofs;
template <int dim, typename NT, size_t nc> using KTCellData = KT::internal::CellData<dim, NT, nc>;
template <int dim, typename NT, size_t nc> using KTNeighborData = KT::internal::NeighborData<dim, NT, nc>;

// nvcc's cudafe emits the first use of a specialization symbolically but constant-folds every
// later one, rendering a size_t argument as the ill-formed functional cast `unsigned long((1))`.
// Naming each specialization once here keeps the use sites below off that path.
using KTCellData_1_1 = KTCellData<1, NumberType, 1>; // <dim, n_components>
using KTCellData_1_2 = KTCellData<1, NumberType, 2>;
using KTCellData_2_1 = KTCellData<2, NumberType, 1>;
using KTNeighborData_1_1 = KTNeighborData<1, NumberType, 1>;
using KTNeighborData_1_2 = KTNeighborData<1, NumberType, 2>;
using KTNeighborData_2_1 = KTNeighborData<2, NumberType, 1>;
using AD = autodiff::Real<1, NumberType>;

TEST_CASE("tag_cell_dofs 1D single component — matching dof is seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 1;

  const KTCellData_1_1 cell_data{
      .x = Point<dim>(1.0),
      .u = {3.5},
      .dof_indices = {42},
  };

  const auto result = tag_cell_dofs(cell_data, 42);

  CHECK(result.x[0] == Catch::Approx(1.0));
  CHECK(val(result.u[0]) == Catch::Approx(3.5));
  CHECK(result.dof_indices[0] == 42);
  CHECK(derivative(result.u[0]) == Catch::Approx(1.0));
}

TEST_CASE("tag_cell_dofs 1D single component — non-matching dof is not seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 1;

  const KTCellData_1_1 cell_data{
      .x = Point<dim>(1.0),
      .u = {3.5},
      .dof_indices = {42},
  };

  const auto result = tag_cell_dofs(cell_data, 99);

  CHECK(val(result.u[0]) == Catch::Approx(3.5));
  CHECK(derivative(result.u[0]) == Catch::Approx(0.0));
}

TEST_CASE("tag_cell_dofs 1D multi-component — only matching component seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 1;

  const KTCellData_1_2 cell_data{
      .x = Point<dim>(0.0),
      .u = {1.0, 2.0},
      .dof_indices = {10, 20},
  };

  SECTION("seed second component")
  {
    const auto result = tag_cell_dofs(cell_data, 20);

    CHECK(val(result.u[0]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[0]) == Catch::Approx(0.0));
    CHECK(val(result.u[1]) == Catch::Approx(2.0));
    CHECK(derivative(result.u[1]) == Catch::Approx(1.0));
    CHECK(result.dof_indices[0] == 10);
    CHECK(result.dof_indices[1] == 20);
  }

  SECTION("seed first component")
  {
    const auto result = tag_cell_dofs(cell_data, 10);

    CHECK(val(result.u[0]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[0]) == Catch::Approx(1.0));
    CHECK(val(result.u[1]) == Catch::Approx(2.0));
    CHECK(derivative(result.u[1]) == Catch::Approx(0.0));
  }
}

TEST_CASE("tag_cell_dofs 2D single component — point preserved and seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 2;

  const KTCellData_2_1 cell_data{
      .x = Point<dim>(0.5, 0.7),
      .u = {4.0},
      .dof_indices = {5},
  };

  const auto result = tag_cell_dofs(cell_data, 5);

  CHECK(result.x[0] == Catch::Approx(0.5));
  CHECK(result.x[1] == Catch::Approx(0.7));
  CHECK(val(result.u[0]) == Catch::Approx(4.0));
  CHECK(derivative(result.u[0]) == Catch::Approx(1.0));
  CHECK(result.dof_indices[0] == 5);
}

// ---------------------------------------------------------------------------
// Tests for make_tagged_neighbors
// ---------------------------------------------------------------------------

TEST_CASE("make_tagged_neighbors 1D single component — matching dof in one face", "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  const KTNeighborData_1_1 nd{
      .x = {Point<dim>(-1.0), Point<dim>(1.0)},
      .u = {std::array<NumberType, nc>{1.0}, std::array<NumberType, nc>{2.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{10},
                      std::array<dealii::types::global_dof_index, nc>{20}},
  };

  const auto result = make_tagged_neighbors(nd, 10);

  // Face 0 matches — seeded
  CHECK(val(result.u[0][0]) == Catch::Approx(1.0));
  CHECK(derivative(result.u[0][0]) == Catch::Approx(1.0));

  // Face 1 does not match — not seeded
  CHECK(val(result.u[1][0]) == Catch::Approx(2.0));
  CHECK(derivative(result.u[1][0]) == Catch::Approx(0.0));

  // Positions preserved
  CHECK(result.x[0][0] == Catch::Approx(-1.0));
  CHECK(result.x[1][0] == Catch::Approx(1.0));

  // dof_indices preserved
  CHECK(result.dof_indices[0][0] == 10);
  CHECK(result.dof_indices[1][0] == 20);
}

TEST_CASE("make_tagged_neighbors 1D single component — no matching dof", "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  const KTNeighborData_1_1 nd{
      .x = {Point<dim>(-1.0), Point<dim>(1.0)},
      .u = {std::array<NumberType, nc>{1.0}, std::array<NumberType, nc>{2.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{10},
                      std::array<dealii::types::global_dof_index, nc>{20}},
  };

  const auto result = make_tagged_neighbors(nd, 99);

  for (size_t face = 0; face < KTNeighborData_1_1::n_faces; ++face)
    for (size_t c = 0; c < nc; ++c)
      CHECK(derivative(result.u[face][c]) == Catch::Approx(0.0));
}

TEST_CASE("make_tagged_neighbors 1D multi-component — specific face and component seeded",
          "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;

  const KTNeighborData_1_2 nd{
      .x = {Point<dim>(-1.0), Point<dim>(1.0)},
      .u = {std::array<NumberType, nc>{1.0, 2.0}, std::array<NumberType, nc>{3.0, 4.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{10, 20},
                      std::array<dealii::types::global_dof_index, nc>{30, 40}},
  };

  SECTION("tag dof 20 — face 0, component 1")
  {
    const auto result = make_tagged_neighbors(nd, 20);

    // Only face 0, component 1 should be seeded
    CHECK(derivative(result.u[0][0]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[0][1]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[1][0]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[1][1]) == Catch::Approx(0.0));

    // Values preserved
    CHECK(val(result.u[0][0]) == Catch::Approx(1.0));
    CHECK(val(result.u[0][1]) == Catch::Approx(2.0));
    CHECK(val(result.u[1][0]) == Catch::Approx(3.0));
    CHECK(val(result.u[1][1]) == Catch::Approx(4.0));
  }

  SECTION("tag dof 30 — face 1, component 0")
  {
    const auto result = make_tagged_neighbors(nd, 30);

    CHECK(derivative(result.u[0][0]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[0][1]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[1][0]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[1][1]) == Catch::Approx(0.0));
  }
}

TEST_CASE("make_tagged_neighbors 2D single component — 4 faces, one matching", "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 2;
  constexpr size_t nc = 1;

  const KTNeighborData_2_1 nd{
      .x = {Point<dim>(-1.0, 0.0), Point<dim>(1.0, 0.0), Point<dim>(0.0, -1.0), Point<dim>(0.0, 1.0)},
      .u = {std::array<NumberType, nc>{1.0}, std::array<NumberType, nc>{2.0}, std::array<NumberType, nc>{3.0},
            std::array<NumberType, nc>{4.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{100},
                      std::array<dealii::types::global_dof_index, nc>{200},
                      std::array<dealii::types::global_dof_index, nc>{300},
                      std::array<dealii::types::global_dof_index, nc>{400}},
  };

  const auto result = make_tagged_neighbors(nd, 300);

  // Only face 2 should be seeded
  CHECK(derivative(result.u[0][0]) == Catch::Approx(0.0));
  CHECK(derivative(result.u[1][0]) == Catch::Approx(0.0));
  CHECK(derivative(result.u[2][0]) == Catch::Approx(1.0));
  CHECK(derivative(result.u[3][0]) == Catch::Approx(0.0));

  // Values preserved
  CHECK(val(result.u[0][0]) == Catch::Approx(1.0));
  CHECK(val(result.u[1][0]) == Catch::Approx(2.0));
  CHECK(val(result.u[2][0]) == Catch::Approx(3.0));
  CHECK(val(result.u[3][0]) == Catch::Approx(4.0));

  // Positions preserved
  CHECK(result.x[0][0] == Catch::Approx(-1.0));
  CHECK(result.x[0][1] == Catch::Approx(0.0));
  CHECK(result.x[2][0] == Catch::Approx(0.0));
  CHECK(result.x[2][1] == Catch::Approx(-1.0));

  // dof_indices preserved
  CHECK(result.dof_indices[2][0] == 300);
}

// -----------------------------------------------------------------------------
// The slots of the source's per-point tuple (TraceBatch::tie).
// -----------------------------------------------------------------------------

/**
 * @brief A source that IS the reconstructed gradient, so the residual reads it back directly.
 *
 * Both fluxes vanish and the mass is excluded (weight_mass = 0), hence residual[cell] == JxW * du/dx
 * as the scheme sees it.
 */
class SourceGradientProbeModel : public DiFfRG::def::AbstractModel<SourceGradientProbeModel, Components>,
                                 public DiFfRG::def::Time,
                                 public DiFfRG::def::LLFFlux<SourceGradientProbeModel>,
                                 public DiFfRG::def::FlowBoundaries<SourceGradientProbeModel>,
                                 public DiFfRG::def::FVDefaultBoundaries<SourceGradientProbeModel>,
                                 public DiFfRG::def::AD<SourceGradientProbeModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.25 * pos[0];
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = NT(0);
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = NT(0);
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    s_i[0] = get<"fe_derivatives">(sol)[0][0];
  }
};

using ExtractorSourceComponents = DiFfRG::ComponentDescriptor<DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u">>,
                                                              DiFfRG::VariableDescriptor<DiFfRG::Scalar<"v">>,
                                                              DiFfRG::ExtractorDescriptor<DiFfRG::Scalar<"e">>>;

/**
 * @brief A source that reads the "extractors" and "variables" slots, with a constant extractor so the
 * expected residual is known without reproducing the EoM search.
 */
class SourceExtractorProbeModel
    : public DiFfRG::def::AbstractModel<SourceExtractorProbeModel, ExtractorSourceComponents>,
      public DiFfRG::def::Time,
      public DiFfRG::def::LLFFlux<SourceExtractorProbeModel>,
      public DiFfRG::def::FlowBoundaries<SourceExtractorProbeModel>,
      public DiFfRG::def::FVDefaultBoundaries<SourceExtractorProbeModel>,
      public DiFfRG::def::AD<SourceExtractorProbeModel>
{
public:
  static constexpr double extractor_value = 2.5;

  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.25 * pos[0];
  }

  template <typename NT, typename Solution>
  void extract(std::array<NT, 1> &extractors, const Point<1> & /*x*/, const Solution & /*sol*/) const
  {
    extractors[0] = NT(extractor_value);
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = NT(0);
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = NT(0);
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    s_i[0] = get<"extractors">(sol)[0] + NT(3.0) * get<"variables">(sol)[0];
  }
};

static double minmod(const double a, const double b)
{
  if (a * b <= 0.0) return 0.0;
  return std::abs(a) < std::abs(b) ? a : b;
}

TEST_CASE("KT source receives the reconstructed cell gradient", "[FV][KT][source][gradient]")
{
  using Model = SourceGradientProbeModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;

  ensure_logger();

  auto json = make_fv_test_json();
  json.set_string("/discretization/grid/x_grid", "0:0.1:1");
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  const auto &dof_handler = discretization.get_dof_handler();
  const double dx = 0.1;
  const unsigned int n_dofs = dof_handler.n_dofs();
  REQUIRE(n_dofs == 10);

  // Cells are addressed by their centre, not by an assumed dof numbering.
  std::vector<dealii::types::global_dof_index> dof_of_cell(n_dofs);
  std::vector<double> center_of_cell(n_dofs);
  std::vector<dealii::types::global_dof_index> cell_dofs(1);
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell->get_dof_indices(cell_dofs);
    const auto index = cell->active_cell_index();
    dof_of_cell[index] = cell_dofs[0];
    center_of_cell[index] = cell->center()[0];
  }

  VectorType sol_dot(n_dofs), residual(n_dofs);

  SECTION("linear field: every one-sided slope agrees, ghosts included")
  {
    const double a = 0.7, b = 1.3;
    VectorType sol(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i)
      sol[dof_of_cell[i]] = a + b * center_of_cell[i];

    assembler.residual(residual, sol, 1.0, sol_dot, 0.0);
    for (unsigned int i = 0; i < n_dofs; ++i)
      CHECK(residual[dof_of_cell[i]] == Catch::Approx(dx * b).margin(1e-12));
  }

  SECTION("curved field: the interior cells see the limited slope")
  {
    VectorType sol(n_dofs);
    std::vector<double> u(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i) {
      const double x = center_of_cell[i];
      u[i] = 0.4 + 0.9 * x - 1.7 * x * x + 2.3 * x * x * x;
      sol[dof_of_cell[i]] = u[i];
    }

    assembler.residual(residual, sol, 1.0, sol_dot, 0.0);
    // Interior only: the boundary cells' slopes involve the ghost stencil, which is what the
    // FD jacobian tests cover.
    for (unsigned int i = 1; i + 1 < n_dofs; ++i) {
      const double expected = dx * minmod((u[i] - u[i - 1]) / dx, (u[i + 1] - u[i]) / dx);
      CHECK(residual[dof_of_cell[i]] == Catch::Approx(expected).margin(1e-12));
    }
  }
}

TEST_CASE("KT source receives extractors and variables", "[FV][KT][source][extractors]")
{
  using Model = SourceExtractorProbeModel;
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;

  ensure_logger();

  auto json = make_fv_test_json();
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  const auto &dof_handler = discretization.get_dof_handler();
  const double dx = 0.25;
  const unsigned int n_dofs = dof_handler.n_dofs();
  REQUIRE(n_dofs == 4);

  VectorType sol(n_dofs), sol_dot(n_dofs), residual(n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i)
    sol[i] = 1.0 + 0.25 * static_cast<double>(i);

  VectorType variables(1);
  variables[0] = 0.75;

  assembler.residual(residual, sol, 1.0, sol_dot, 0.0, variables);

  const double expected = dx * (Model::extractor_value + 3.0 * variables[0]);
  for (unsigned int i = 0; i < n_dofs; ++i)
    CHECK(residual[i] == Catch::Approx(expected).margin(1e-12));
}

/**
 * @brief A flux and a diffusion flux that read the "extractors" and "variables" slots.
 *
 * The point of the pair below is that FluxHardcodedProbeModel is the same model with the two
 * numbers written in, so residual and jacobian have to agree entry for entry. That makes the check
 * independent of the KT reconstruction: it only asserts that what reaches flux()/diffusion_flux()
 * through the tuple is what extract() and the variables vector put there.
 */
template <bool ReadFromTuple>
class FluxExtractorProbeModelT
    : public DiFfRG::def::AbstractModel<FluxExtractorProbeModelT<ReadFromTuple>, ExtractorSourceComponents>,
      public DiFfRG::def::Time,
      public DiFfRG::def::LLFFlux<FluxExtractorProbeModelT<ReadFromTuple>>,
      public DiFfRG::def::FlowBoundaries<FluxExtractorProbeModelT<ReadFromTuple>>,
      public DiFfRG::def::FVDefaultBoundaries<FluxExtractorProbeModelT<ReadFromTuple>>,
      public DiFfRG::def::AD<FluxExtractorProbeModelT<ReadFromTuple>>
{
public:
  static constexpr double extractor_value = 2.5;
  static constexpr double variable_value = 0.75;

  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.25 * pos[0];
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = variable_value; }

  template <typename NT, typename Solution>
  void extract(std::array<NT, 1> &extractors, const Point<1> & /*x*/, const Solution & /*sol*/) const
  {
    extractors[0] = NT(extractor_value);
  }

  /// The advection speed, either read out of the tuple or written in.
  template <typename Solution> double speed(const Solution &sol) const
  {
    if constexpr (ReadFromTuple)
      return get<"extractors">(sol)[0] + 3.0 * get<"variables">(sol)[0];
    else
      return extractor_value + 3.0 * variable_value;
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    F_i[0][0] = NT(speed(sol)) * get<"fe_functions">(sol)[0];
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    F_i[0][0] = NT(-0.1 * speed(sol)) * get<"fe_derivatives">(sol)[0][0];
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    s_i[0] = NT(0);
  }
};

using FluxExtractorProbeModel = FluxExtractorProbeModelT<true>;
using FluxHardcodedProbeModel = FluxExtractorProbeModelT<false>;

/// Assemble residual and jacobian for one of the two probe models above.
template <typename Model>
static void assemble_flux_probe(VectorType &residual, dealii::SparsityPattern &sp,
                                dealii::SparseMatrix<NumberType> &jacobian)
{
  using Discretization = DiFfRG::FV::Discretization<Model, DiFfRG::RectangularMeshSerial<1>, NumberType>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;

  auto json = make_fv_test_json();
  Model model;
  DiFfRG::RectangularMeshSerial<1> mesh{DiFfRG::Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  const unsigned int n_dofs = discretization.get_dof_handler().n_dofs();
  VectorType sol(n_dofs), sol_dot(n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i)
    sol[i] = 1.0 + 0.25 * static_cast<double>(i);

  VectorType variables(1);
  variables[0] = Model::variable_value;

  residual.reinit(n_dofs);
  assembler.residual(residual, sol, 1.0, sol_dot, 0.0, variables);

  sp.copy_from(assembler.get_sparsity_pattern_jacobian());
  jacobian.reinit(sp);
  assembler.jacobian(jacobian, sol, 1.0, sol_dot, 0.0, 0.0, variables);
}

TEST_CASE("KT flux and diffusion_flux receive extractors and variables", "[FV][KT][flux][extractors]")
{
  ensure_logger();

  VectorType r_tuple, r_hard;
  dealii::SparsityPattern sp_tuple, sp_hard;
  dealii::SparseMatrix<NumberType> j_tuple, j_hard;

  assemble_flux_probe<FluxExtractorProbeModel>(r_tuple, sp_tuple, j_tuple);
  assemble_flux_probe<FluxHardcodedProbeModel>(r_hard, sp_hard, j_hard);

  REQUIRE(r_tuple.size() == r_hard.size());
  // A zero residual would pass this trivially, so make sure the flux actually moved something.
  REQUIRE(r_hard.l2_norm() > 1e-6);
  for (unsigned int i = 0; i < r_tuple.size(); ++i)
    CHECK(r_tuple[i] == Catch::Approx(r_hard[i]).margin(1e-14));

  // The jacobian is assembled through a separate AD path, so it has to be checked separately: a
  // thread-through that reaches residual() but not jacobian() would pass the check above.
  REQUIRE(j_tuple.n_nonzero_elements() == j_hard.n_nonzero_elements());
  REQUIRE(j_hard.frobenius_norm() > 1e-6);
  for (unsigned int row = 0; row < j_tuple.m(); ++row)
    for (auto it = j_tuple.begin(row); it != j_tuple.end(row); ++it)
      CHECK(it->value() == Catch::Approx(j_hard(row, it->column())).margin(1e-14));
}
