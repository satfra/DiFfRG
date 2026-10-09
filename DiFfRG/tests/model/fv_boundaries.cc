#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <autodiff/forward/real/real.hpp>

#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/model/fv_boundaries.hh>

#include <random>
#include <sstream>
#include <type_traits>

namespace
{
  using namespace dealii;
  namespace KT = DiFfRG::FV::KurganovTadmor;

  constexpr double tolerance = 1e-12;
  constexpr double mismatch_tolerance = 1e-9;
  constexpr double limiter_margin = 0.15;

  using Reconstructor = DiFfRG::def::TVDReconstructor<1, DiFfRG::def::MinModLimiter, double>;

  template <typename T> using FaceArray = std::array<T, GeometryInfo<1>::faces_per_cell>;

  struct DummyModelDefault : public DiFfRG::def::FVDefaultBoundaries<DummyModelDefault>
  {
  };

  struct DummyModelRhoSymmetric
      : public DiFfRG::def::RhoSymmetricLinearExtrapolationBoundaries<DummyModelRhoSymmetric>
  {
  };

	  struct DummyModelOriginOddLinearExtrapolation
	      : public DiFfRG::def::OriginOddLinearExtrapolationBoundaries<DummyModelOriginOddLinearExtrapolation>
	  {
	  };

	  struct DummyModelOriginShiftedOddLinearExtrapolation
	      : public DiFfRG::def::OriginShiftedOddLinearExtrapolationBoundaries<DummyModelOriginShiftedOddLinearExtrapolation>
	  {
	    static constexpr double origin_value = -3.0;

	    template <typename NumberType, std::size_t n_components>
	    std::array<NumberType, n_components> origin_odd_reflection_values() const
	    {
	      static_assert(n_components == 1, "Dummy shifted boundary model is scalar.");
	      return {NumberType(origin_value)};
	    }
	  };

  template <int dim> struct TaggedBoundaryStencilFactory;

  template <> struct TaggedBoundaryStencilFactory<1>
  {
    template <std::size_t n_components>
    static DiFfRG::def::BoundaryStencilValues<1, autodiff::Real<1, double>, n_components>
    make(const DiFfRG::def::BoundaryStencilValues<1, double, n_components> &u_stencil,
         const std::array<std::array<types::global_dof_index, n_components>, 5> &dof_stencil,
         const types::global_dof_index dof_j)
    {
      DiFfRG::def::BoundaryStencilValues<1, autodiff::Real<1, double>, n_components> result{};
      for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t c = 0; c < n_components; ++c) {
          result[i][c] = autodiff::Real<1, double>(u_stencil[i][c]);
          if (dof_stencil[i][c] == dof_j) autodiff::detail::seed<1>(result[i][c], 1.0);
        }
      }
      return result;
    }
  };

  template <> struct TaggedBoundaryStencilFactory<2>
  {
    template <std::size_t n_components>
    static DiFfRG::def::BoundaryStencilValues<2, autodiff::Real<1, double>, n_components>
    make(const DiFfRG::def::BoundaryStencilValues<2, double, n_components> &u_stencil,
         const std::array<std::array<types::global_dof_index, n_components>, 7> &dof_stencil,
         const types::global_dof_index dof_j)
    {
      DiFfRG::def::BoundaryStencilValues<2, autodiff::Real<1, double>, n_components> result{};
      for (std::size_t i = 0; i < 7; ++i) {
        for (std::size_t c = 0; c < n_components; ++c) {
          result[i][c] = autodiff::Real<1, double>(u_stencil[i][c]);
          if (dof_stencil[i][c] == dof_j) autodiff::detail::seed<1>(result[i][c], 1.0);
        }
      }
      return result;
    }
  };

  template <int dim, std::size_t n_components>
  DiFfRG::def::BoundaryStencilValues<dim, autodiff::Real<1, double>, n_components>
  make_tagged_stencil(const DiFfRG::def::BoundaryStencilValues<dim, double, n_components> &u_stencil,
                      const std::array<std::array<types::global_dof_index, n_components>, 2 * dim + 3> &dof_stencil,
                      const types::global_dof_index dof_j)
  {
    return TaggedBoundaryStencilFactory<dim>::template make<n_components>(u_stencil, dof_stencil, dof_j);
  }

  struct ReconstructionSummary1D {
    double u_plus = 0.0;
    double u_minus = 0.0;
    double grad_u_plus = 0.0;
    double grad_u_minus = 0.0;
    std::array<double, 3> du_plus{};
    std::array<double, 3> du_minus{};
    std::array<double, 3> dgrad_u_plus{};
    std::array<double, 3> dgrad_u_minus{};
  };

  template <typename NumberType> struct BoundaryTrace1D {
    NumberType u_plus{};
    NumberType u_minus{};
    NumberType grad_u_plus{};
    NumberType grad_u_minus{};
  };

  template <typename NumberType> double scalar_value(const NumberType &value)
  {
    if constexpr (std::is_arithmetic_v<NumberType>)
      return value;
    else
      return val(value);
  }

  std::string describe_reconstruction(const ReconstructionSummary1D &reconstruction)
  {
    std::ostringstream out;
    out << "u_plus=" << reconstruction.u_plus << ", u_minus=" << reconstruction.u_minus
        << ", grad_u_plus=" << reconstruction.grad_u_plus << ", grad_u_minus=" << reconstruction.grad_u_minus;
    return out.str();
  }

  template <typename NumberType>
  BoundaryTrace1D<NumberType> compute_boundary_trace(
      const DiFfRG::def::BoundaryStencilPoints<1> &x_stencil,
      const DiFfRG::def::BoundaryStencilValues<1, NumberType, 1> &u_stencil,
      const std::array<std::array<types::global_dof_index, 1>, 5> &dof_stencil, const Point<1> &x_face)
  {
    constexpr int dim = 1;
    using BoundaryIndex = DiFfRG::FV::KurganovTadmor::internal::BoundaryStencilIndex<dim>;
    using ReconstructorNumber =
        DiFfRG::def::TVDReconstructor<dim, DiFfRG::def::MinModLimiter, NumberType>;

    KT::internal::BoundaryStencilData<dim, NumberType, 1> boundary_stencil{};
    boundary_stencil.x = x_stencil;
    boundary_stencil.u = u_stencil;
    boundary_stencil.dof_indices = dof_stencil;
    boundary_stencil.lower_boundary = KT::internal::is_lower_boundary_stencil<dim, NumberType, 1>(x_stencil, x_face);
    boundary_stencil.ghost_center =
        boundary_stencil.lower_boundary ? BoundaryIndex::lower_inner : BoundaryIndex::upper_inner;
    boundary_stencil.ghost_left =
        boundary_stencil.lower_boundary ? BoundaryIndex::lower_outer : BoundaryIndex::physical_cell;
    boundary_stencil.ghost_right =
        boundary_stencil.lower_boundary ? BoundaryIndex::physical_cell : BoundaryIndex::upper_outer;
    const auto physical_stencil =
        KT::internal::make_physical_boundary_side_stencil<dim, NumberType, 1>(boundary_stencil);
    const auto ghost_stencil = KT::internal::make_ghost_boundary_side_stencil<dim, NumberType, 1>(boundary_stencil);

    const auto u_grad_physical = ReconstructorNumber::template compute_gradient<1>(
        physical_stencil.cell.x, physical_stencil.cell.u, physical_stencil.neighbors.x, physical_stencil.neighbors.u);
    const auto u_grad_ghost = ReconstructorNumber::template compute_gradient<1>(
        ghost_stencil.cell.x, ghost_stencil.cell.u, ghost_stencil.neighbors.x, ghost_stencil.neighbors.u);
    const auto u_grad_minus = ReconstructorNumber::template compute_gradient_at_point<1>(
        physical_stencil.cell.x, x_face, physical_stencil.cell.u, physical_stencil.neighbors.x,
        physical_stencil.neighbors.u);
    const auto u_grad_plus = ReconstructorNumber::template compute_gradient_at_point<1>(
        ghost_stencil.cell.x, x_face, ghost_stencil.cell.u, ghost_stencil.neighbors.x, ghost_stencil.neighbors.u);
    const auto u_minus =
        KT::internal::reconstruct_u<1, NumberType, 1>(physical_stencil.cell.u, physical_stencil.cell.x, x_face,
                                                     u_grad_physical);
    const auto u_plus =
        KT::internal::reconstruct_u<1, NumberType, 1>(ghost_stencil.cell.u, ghost_stencil.cell.x, x_face, u_grad_ghost);

    BoundaryTrace1D<NumberType> result;
    result.u_plus = u_plus[0];
    result.u_minus = u_minus[0];
    result.grad_u_plus = u_grad_plus[0][0];
    result.grad_u_minus = u_grad_minus[0][0];
    return result;
  }

  bool differs(const ReconstructionSummary1D &lhs, const ReconstructionSummary1D &rhs,
               const double eps = mismatch_tolerance)
  {
    const auto differs_scalar = [eps](const double a, const double b) { return std::abs(a - b) > eps; };
    if (differs_scalar(lhs.u_plus, rhs.u_plus) || differs_scalar(lhs.u_minus, rhs.u_minus) ||
        differs_scalar(lhs.grad_u_plus, rhs.grad_u_plus) || differs_scalar(lhs.grad_u_minus, rhs.grad_u_minus))
      return true;

    for (std::size_t i = 0; i < 3; ++i) {
      if (differs_scalar(lhs.du_plus[i], rhs.du_plus[i]) || differs_scalar(lhs.du_minus[i], rhs.du_minus[i]) ||
          differs_scalar(lhs.dgrad_u_plus[i], rhs.dgrad_u_plus[i]) ||
          differs_scalar(lhs.dgrad_u_minus[i], rhs.dgrad_u_minus[i]))
        return true;
    }

    return false;
  }

	  ReconstructionSummary1D lower_boundary_half_domain_reconstruction(const std::array<double, 3> &physical_values)
  {
    DummyModelOriginOddLinearExtrapolation model;

    const DiFfRG::def::BoundaryStencilValues<1, double, 1> raw_u_stencil{
        {{{0.0}}, {{0.0}}, {{physical_values[0]}}, {{physical_values[1]}}, {{physical_values[2]}}}};
    const DiFfRG::def::BoundaryStencilPoints<1> raw_x_stencil{
        {Point<1>(0.0), Point<1>(0.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};
    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
        {{{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}, {{10}}, {{11}}, {{12}}}};

    auto u_stencil = raw_u_stencil;
    auto x_stencil = raw_x_stencil;
    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(0.0)));

    const auto reconstruction = compute_boundary_trace(x_stencil, u_stencil, dof_stencil, Point<1>(0.0));

    ReconstructionSummary1D result;
    result.u_plus = scalar_value(reconstruction.u_plus);
    result.u_minus = scalar_value(reconstruction.u_minus);
    result.grad_u_plus = scalar_value(reconstruction.grad_u_plus);
    result.grad_u_minus = scalar_value(reconstruction.grad_u_minus);

    for (std::size_t derivative_index = 0; derivative_index < 3; ++derivative_index) {
      auto tagged_stencil = make_tagged_stencil<1>(raw_u_stencil, dof_stencil, 10 + derivative_index);
      auto tagged_points = raw_x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, tagged_points, Point<1>(0.0)));
      const auto tagged_reconstruction = compute_boundary_trace(tagged_points, tagged_stencil, dof_stencil, Point<1>(0.0));
      result.du_plus[derivative_index] = derivative(tagged_reconstruction.u_plus);
      result.du_minus[derivative_index] = derivative(tagged_reconstruction.u_minus);
      result.dgrad_u_plus[derivative_index] = derivative(tagged_reconstruction.grad_u_plus);
      result.dgrad_u_minus[derivative_index] = derivative(tagged_reconstruction.grad_u_minus);
    }

	    return result;
	  }

	  ReconstructionSummary1D shifted_lower_boundary_half_domain_reconstruction(const std::array<double, 3> &physical_values)
	  {
	    DummyModelOriginShiftedOddLinearExtrapolation model;

	    const DiFfRG::def::BoundaryStencilValues<1, double, 1> raw_u_stencil{
	        {{{0.0}}, {{0.0}}, {{physical_values[0]}}, {{physical_values[1]}}, {{physical_values[2]}}}};
	    const DiFfRG::def::BoundaryStencilPoints<1> raw_x_stencil{
	        {Point<1>(0.0), Point<1>(0.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};
	    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
	        {{{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}, {{10}}, {{11}}, {{12}}}};

	    auto u_stencil = raw_u_stencil;
	    auto x_stencil = raw_x_stencil;
	    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(0.0)));

	    const auto reconstruction = compute_boundary_trace(x_stencil, u_stencil, dof_stencil, Point<1>(0.0));

	    ReconstructionSummary1D result;
	    result.u_plus = scalar_value(reconstruction.u_plus);
	    result.u_minus = scalar_value(reconstruction.u_minus);
	    result.grad_u_plus = scalar_value(reconstruction.grad_u_plus);
	    result.grad_u_minus = scalar_value(reconstruction.grad_u_minus);

	    for (std::size_t derivative_index = 0; derivative_index < 3; ++derivative_index) {
	      auto tagged_stencil = make_tagged_stencil<1>(raw_u_stencil, dof_stencil, 10 + derivative_index);
	      auto tagged_points = raw_x_stencil;
	      REQUIRE(model.apply_boundary_stencil(tagged_stencil, tagged_points, Point<1>(0.0)));
	      const auto tagged_reconstruction = compute_boundary_trace(tagged_points, tagged_stencil, dof_stencil, Point<1>(0.0));
	      result.du_plus[derivative_index] = derivative(tagged_reconstruction.u_plus);
	      result.du_minus[derivative_index] = derivative(tagged_reconstruction.u_minus);
	      result.dgrad_u_plus[derivative_index] = derivative(tagged_reconstruction.grad_u_plus);
	      result.dgrad_u_minus[derivative_index] = derivative(tagged_reconstruction.grad_u_minus);
	    }

	    return result;
	  }

  ReconstructionSummary1D rho_lower_boundary_half_domain_reconstruction(const std::array<double, 3> &physical_values)
  {
    DummyModelRhoSymmetric model;

    const DiFfRG::def::BoundaryStencilValues<1, double, 1> raw_u_stencil{
        {{{0.0}}, {{0.0}}, {{physical_values[0]}}, {{physical_values[1]}}, {{physical_values[2]}}}};
    const DiFfRG::def::BoundaryStencilPoints<1> raw_x_stencil{
        {Point<1>(0.0), Point<1>(0.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};
    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
        {{{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}, {{10}}, {{11}}, {{12}}}};

    auto u_stencil = raw_u_stencil;
    auto x_stencil = raw_x_stencil;
    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(0.0)));

    const auto reconstruction = compute_boundary_trace(x_stencil, u_stencil, dof_stencil, Point<1>(0.0));

    ReconstructionSummary1D result;
    result.u_plus = scalar_value(reconstruction.u_plus);
    result.u_minus = scalar_value(reconstruction.u_minus);
    result.grad_u_plus = scalar_value(reconstruction.grad_u_plus);
    result.grad_u_minus = scalar_value(reconstruction.grad_u_minus);

    for (std::size_t derivative_index = 0; derivative_index < 3; ++derivative_index) {
      auto tagged_stencil = make_tagged_stencil<1>(raw_u_stencil, dof_stencil, 10 + derivative_index);
      auto tagged_points = raw_x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, tagged_points, Point<1>(0.0)));
      const auto tagged_reconstruction = compute_boundary_trace(tagged_points, tagged_stencil, dof_stencil, Point<1>(0.0));
      result.du_plus[derivative_index] = derivative(tagged_reconstruction.u_plus);
      result.du_minus[derivative_index] = derivative(tagged_reconstruction.u_minus);
      result.dgrad_u_plus[derivative_index] = derivative(tagged_reconstruction.grad_u_plus);
      result.dgrad_u_minus[derivative_index] = derivative(tagged_reconstruction.grad_u_minus);
    }

    return result;
  }

  ReconstructionSummary1D origin_full_domain_interior_face_reconstruction(const std::array<double, 3> &physical_values)
  {
    ReconstructionSummary1D result;

    const Point<1> x_face(0.0);
    const Point<1> right_center(0.5);
    const Point<1> left_center(-0.5);

    const std::array<Point<1>, 2> x_neighbors_right{{Point<1>(-0.5), Point<1>(1.5)}};
    const std::array<Point<1>, 2> x_neighbors_left{{Point<1>(-1.5), Point<1>(0.5)}};

    const std::array<double, 1> u_center_right{{physical_values[0]}};
    const std::array<double, 1> u_center_left{{-physical_values[0]}};
    const std::array<std::array<double, 1>, 2> u_neighbors_right{{{{-physical_values[0]}}, {{physical_values[1]}}}};
    const std::array<std::array<double, 1>, 2> u_neighbors_left{{{{-physical_values[1]}}, {{physical_values[0]}}}};

    const auto u_grad_right = Reconstructor::compute_gradient<1>(right_center, u_center_right, x_neighbors_right,
                                                                 u_neighbors_right);
    const auto u_grad_left =
        Reconstructor::compute_gradient<1>(left_center, u_center_left, x_neighbors_left, u_neighbors_left);
    const auto u_grad_minus =
        Reconstructor::compute_gradient_at_point<1>(right_center, x_face, u_center_right, x_neighbors_right, u_neighbors_right);
    const auto u_grad_plus =
        Reconstructor::compute_gradient_at_point<1>(left_center, x_face, u_center_left, x_neighbors_left, u_neighbors_left);

    result.u_minus = KT::internal::reconstruct_u<1, double, 1>(u_center_right, right_center, x_face, u_grad_right)[0];
    result.u_plus = KT::internal::reconstruct_u<1, double, 1>(u_center_left, left_center, x_face, u_grad_left)[0];
    result.grad_u_minus = u_grad_minus[0][0];
    result.grad_u_plus = u_grad_plus[0][0];

    for (std::size_t derivative_index = 0; derivative_index < 3; ++derivative_index) {
      using AD = autodiff::Real<1, double>;
      std::array<AD, 3> tagged_physical{AD(physical_values[0]), AD(physical_values[1]), AD(physical_values[2])};
      autodiff::detail::seed<1>(tagged_physical[derivative_index], 1.0);

      const std::array<AD, 1> tagged_center_right{{tagged_physical[0]}};
      const std::array<AD, 1> tagged_center_left{{-tagged_physical[0]}};
      const std::array<std::array<AD, 1>, 2> tagged_neighbors_right{{{{-tagged_physical[0]}}, {{tagged_physical[1]}}}};
      const std::array<std::array<AD, 1>, 2> tagged_neighbors_left{{{{-tagged_physical[1]}}, {{tagged_physical[0]}}}};

      using ReconstructorAD = DiFfRG::def::TVDReconstructor<1, DiFfRG::def::MinModLimiter, AD>;
      const auto tagged_u_grad_right = ReconstructorAD::compute_gradient<1>(right_center, tagged_center_right,
                                                                            x_neighbors_right, tagged_neighbors_right);
      const auto tagged_u_grad_left =
          ReconstructorAD::compute_gradient<1>(left_center, tagged_center_left, x_neighbors_left, tagged_neighbors_left);
      const auto tagged_u_grad_minus = ReconstructorAD::compute_gradient_at_point<1>(
          right_center, x_face, tagged_center_right, x_neighbors_right, tagged_neighbors_right);
      const auto tagged_u_grad_plus = ReconstructorAD::compute_gradient_at_point<1>(
          left_center, x_face, tagged_center_left, x_neighbors_left, tagged_neighbors_left);
      const auto tagged_u_minus =
          KT::internal::reconstruct_u<1, AD, 1>(tagged_center_right, right_center, x_face, tagged_u_grad_right);
      const auto tagged_u_plus =
          KT::internal::reconstruct_u<1, AD, 1>(tagged_center_left, left_center, x_face, tagged_u_grad_left);

      result.du_minus[derivative_index] = derivative(tagged_u_minus[0]);
      result.du_plus[derivative_index] = derivative(tagged_u_plus[0]);
      result.dgrad_u_minus[derivative_index] = derivative(tagged_u_grad_minus[0][0]);
      result.dgrad_u_plus[derivative_index] = derivative(tagged_u_grad_plus[0][0]);
    }

    return result;
  }

  ReconstructionSummary1D rho_full_domain_interior_face_reconstruction(const std::array<double, 3> &physical_values)
  {
    ReconstructionSummary1D result;

    const Point<1> x_face(0.0);
    const Point<1> right_center(0.5);
    const Point<1> left_center(-0.5);

    const std::array<Point<1>, 2> x_neighbors_right{{Point<1>(-0.5), Point<1>(1.5)}};
    const std::array<Point<1>, 2> x_neighbors_left{{Point<1>(-1.5), Point<1>(0.5)}};

    const std::array<double, 1> u_center_right{{physical_values[0]}};
    const std::array<double, 1> u_center_left{{physical_values[0]}};
    const std::array<std::array<double, 1>, 2> u_neighbors_right{{{{physical_values[0]}}, {{physical_values[1]}}}};
    const std::array<std::array<double, 1>, 2> u_neighbors_left{{{{physical_values[1]}}, {{physical_values[0]}}}};

    const auto u_grad_right = Reconstructor::compute_gradient<1>(right_center, u_center_right, x_neighbors_right,
                                                                 u_neighbors_right);
    const auto u_grad_left =
        Reconstructor::compute_gradient<1>(left_center, u_center_left, x_neighbors_left, u_neighbors_left);
    const auto u_grad_minus =
        Reconstructor::compute_gradient_at_point<1>(right_center, x_face, u_center_right, x_neighbors_right, u_neighbors_right);
    const auto u_grad_plus =
        Reconstructor::compute_gradient_at_point<1>(left_center, x_face, u_center_left, x_neighbors_left, u_neighbors_left);

    result.u_minus = KT::internal::reconstruct_u<1, double, 1>(u_center_right, right_center, x_face, u_grad_right)[0];
    result.u_plus = KT::internal::reconstruct_u<1, double, 1>(u_center_left, left_center, x_face, u_grad_left)[0];
    result.grad_u_minus = u_grad_minus[0][0];
    result.grad_u_plus = u_grad_plus[0][0];

    for (std::size_t derivative_index = 0; derivative_index < 3; ++derivative_index) {
      using AD = autodiff::Real<1, double>;
      std::array<AD, 3> tagged_physical{AD(physical_values[0]), AD(physical_values[1]), AD(physical_values[2])};
      autodiff::detail::seed<1>(tagged_physical[derivative_index], 1.0);

      const std::array<AD, 1> tagged_center_right{{tagged_physical[0]}};
      const std::array<AD, 1> tagged_center_left{{tagged_physical[0]}};
      const std::array<std::array<AD, 1>, 2> tagged_neighbors_right{{{{tagged_physical[0]}}, {{tagged_physical[1]}}}};
      const std::array<std::array<AD, 1>, 2> tagged_neighbors_left{{{{tagged_physical[1]}}, {{tagged_physical[0]}}}};

      using ReconstructorAD = DiFfRG::def::TVDReconstructor<1, DiFfRG::def::MinModLimiter, AD>;
      const auto tagged_u_grad_right = ReconstructorAD::compute_gradient<1>(right_center, tagged_center_right,
                                                                            x_neighbors_right, tagged_neighbors_right);
      const auto tagged_u_grad_left =
          ReconstructorAD::compute_gradient<1>(left_center, tagged_center_left, x_neighbors_left, tagged_neighbors_left);
      const auto tagged_u_grad_minus = ReconstructorAD::compute_gradient_at_point<1>(
          right_center, x_face, tagged_center_right, x_neighbors_right, tagged_neighbors_right);
      const auto tagged_u_grad_plus = ReconstructorAD::compute_gradient_at_point<1>(
          left_center, x_face, tagged_center_left, x_neighbors_left, tagged_neighbors_left);
      const auto tagged_u_minus =
          KT::internal::reconstruct_u<1, AD, 1>(tagged_center_right, right_center, x_face, tagged_u_grad_right);
      const auto tagged_u_plus =
          KT::internal::reconstruct_u<1, AD, 1>(tagged_center_left, left_center, x_face, tagged_u_grad_left);

      result.du_minus[derivative_index] = derivative(tagged_u_minus[0]);
      result.du_plus[derivative_index] = derivative(tagged_u_plus[0]);
      result.dgrad_u_minus[derivative_index] = derivative(tagged_u_grad_minus[0][0]);
      result.dgrad_u_plus[derivative_index] = derivative(tagged_u_grad_plus[0][0]);
    }

    return result;
  }

  ReconstructionSummary1D upper_boundary_half_domain_reconstruction(const std::array<double, 3> &physical_values)
  {
    DummyModelOriginOddLinearExtrapolation model;

    const DiFfRG::def::BoundaryStencilValues<1, double, 1> raw_u_stencil{
        {{{physical_values[0]}}, {{physical_values[1]}}, {{physical_values[2]}}, {{0.0}}, {{0.0}}}};
    const DiFfRG::def::BoundaryStencilPoints<1> raw_x_stencil{
        {Point<1>(7.5), Point<1>(8.5), Point<1>(9.5), Point<1>(0.0), Point<1>(0.0)}};
    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
        {{{20}}, {{21}}, {{22}}, {{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}}};

    auto u_stencil = raw_u_stencil;
    auto x_stencil = raw_x_stencil;
    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(10.0)));

    const auto reconstruction = compute_boundary_trace(x_stencil, u_stencil, dof_stencil, Point<1>(10.0));

    ReconstructionSummary1D result;
    result.u_plus = scalar_value(reconstruction.u_plus);
    result.u_minus = scalar_value(reconstruction.u_minus);
    result.grad_u_plus = scalar_value(reconstruction.grad_u_plus);
    result.grad_u_minus = scalar_value(reconstruction.grad_u_minus);

    for (std::size_t derivative_index = 0; derivative_index < 3; ++derivative_index) {
      auto tagged_stencil = make_tagged_stencil<1>(raw_u_stencil, dof_stencil, 20 + derivative_index);
      auto tagged_points = raw_x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, tagged_points, Point<1>(10.0)));
      const auto tagged_reconstruction = compute_boundary_trace(tagged_points, tagged_stencil, dof_stencil, Point<1>(10.0));
      result.du_plus[derivative_index] = derivative(tagged_reconstruction.u_plus);
      result.du_minus[derivative_index] = derivative(tagged_reconstruction.u_minus);
      result.dgrad_u_plus[derivative_index] = derivative(tagged_reconstruction.grad_u_plus);
      result.dgrad_u_minus[derivative_index] = derivative(tagged_reconstruction.grad_u_minus);
    }

    return result;
  }

  bool origin_random_trial_is_stable(const std::array<double, 3> &physical_values)
  {
    const double p0 = physical_values[0];
    const double p1 = physical_values[1];
    const double p2 = physical_values[2];

    const double full_right_left = 2.0 * p0;
    const double full_right_right = p1 - p0;
    const double full_left_left = p0 - p1;
    const double full_left_right = 2.0 * p0;
    const double half_ghost_left = p1 - p2;
    const double half_ghost_right = p1;

    return std::abs(full_right_left) > limiter_margin && std::abs(full_right_right) > limiter_margin &&
           std::abs(full_left_left) > limiter_margin && std::abs(full_left_right) > limiter_margin &&
           std::abs(half_ghost_left) > limiter_margin && std::abs(half_ghost_right) > limiter_margin &&
           std::abs(std::abs(full_right_left) - std::abs(full_right_right)) > limiter_margin &&
           std::abs(std::abs(full_left_left) - std::abs(full_left_right)) > limiter_margin &&
           std::abs(std::abs(half_ghost_left) - std::abs(half_ghost_right)) > limiter_margin;
  }

  std::array<double, 3> draw_random_values(std::mt19937 &rng)
  {
    std::uniform_real_distribution<double> distribution(-3.0, 3.0);
    return {distribution(rng), distribution(rng), distribution(rng)};
  }
} // namespace

TEST_CASE("FVDefaultBoundaries applies affine two-ghost-cell stencils", "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelDefault model;

  SECTION("lower boundary extrapolates affinely from the first two physical cells")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{11.0}}, {{12.0}}, {{1.0}}, {{2.0}}, {{4.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(-2.0), Point<1>(-1.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(0.0)));

    CHECK_THAT(x_stencil[0][0], Catch::Matchers::WithinAbs(-1.5, tolerance));
    CHECK_THAT(x_stencil[1][0], Catch::Matchers::WithinAbs(-0.5, tolerance));
    CHECK_THAT(u_stencil[0][0], Catch::Matchers::WithinAbs(-1.0, tolerance));
    CHECK_THAT(u_stencil[1][0], Catch::Matchers::WithinAbs(0.0, tolerance));
  }

  SECTION("upper boundary extrapolates affinely from the last two physical cells")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{1.0}}, {{3.0}}, {{5.0}}, {{17.0}}, {{19.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(7.5), Point<1>(8.5), Point<1>(9.5), Point<1>(20.0), Point<1>(21.0)}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(10.0)));

    CHECK_THAT(x_stencil[3][0], Catch::Matchers::WithinAbs(10.5, tolerance));
    CHECK_THAT(x_stencil[4][0], Catch::Matchers::WithinAbs(11.5, tolerance));
    CHECK_THAT(u_stencil[3][0], Catch::Matchers::WithinAbs(7.0, tolerance));
    CHECK_THAT(u_stencil[4][0], Catch::Matchers::WithinAbs(9.0, tolerance));
  }
}

TEST_CASE("FVDefaultBoundaries applies affine two-ghost-cell stencils in 2D", "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelDefault model;

  SECTION("lower y boundary extrapolates along y and preserves x")
  {
    def::BoundaryStencilValues<2, double, 2> u_stencil{
        {{{11.0, 21.0}}, {{12.0, 22.0}}, {{1.0, 4.0}}, {{2.0, 6.0}}, {{4.0, 10.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}};
    def::BoundaryStencilPoints<2> x_stencil{{Point<2>(8.0, 8.0), Point<2>(9.0, 9.0), Point<2>(3.0, 0.5),
                                             Point<2>(3.0, 1.5), Point<2>(3.0, 2.5), Point<2>(), Point<2>()}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<2>(3.0, 0.0)));

    CHECK_THAT(x_stencil[0][0], Catch::Matchers::WithinAbs(3.0, tolerance));
    CHECK_THAT(x_stencil[1][0], Catch::Matchers::WithinAbs(3.0, tolerance));
    CHECK_THAT(x_stencil[0][1], Catch::Matchers::WithinAbs(-1.5, tolerance));
    CHECK_THAT(x_stencil[1][1], Catch::Matchers::WithinAbs(-0.5, tolerance));
    CHECK_THAT(u_stencil[0][0], Catch::Matchers::WithinAbs(-1.0, tolerance));
    CHECK_THAT(u_stencil[1][0], Catch::Matchers::WithinAbs(0.0, tolerance));
    CHECK_THAT(u_stencil[0][1], Catch::Matchers::WithinAbs(0.0, tolerance));
    CHECK_THAT(u_stencil[1][1], Catch::Matchers::WithinAbs(2.0, tolerance));
  }

  SECTION("upper x boundary extrapolates along x and preserves y")
  {
    def::BoundaryStencilValues<2, double, 2> u_stencil{
        {{{1.0, 4.0}}, {{3.0, 8.0}}, {{5.0, 12.0}}, {{17.0, 18.0}}, {{19.0, 20.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}};
    def::BoundaryStencilPoints<2> x_stencil{{Point<2>(7.5, 2.0), Point<2>(8.5, 2.0), Point<2>(9.5, 2.0),
                                             Point<2>(20.0, 21.0), Point<2>(21.0, 22.0), Point<2>(), Point<2>()}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<2>(10.0, 2.0)));

    CHECK_THAT(x_stencil[3][0], Catch::Matchers::WithinAbs(10.5, tolerance));
    CHECK_THAT(x_stencil[4][0], Catch::Matchers::WithinAbs(11.5, tolerance));
    CHECK_THAT(x_stencil[3][1], Catch::Matchers::WithinAbs(2.0, tolerance));
    CHECK_THAT(x_stencil[4][1], Catch::Matchers::WithinAbs(2.0, tolerance));
    CHECK_THAT(u_stencil[3][0], Catch::Matchers::WithinAbs(7.0, tolerance));
    CHECK_THAT(u_stencil[4][0], Catch::Matchers::WithinAbs(9.0, tolerance));
    CHECK_THAT(u_stencil[3][1], Catch::Matchers::WithinAbs(16.0, tolerance));
    CHECK_THAT(u_stencil[4][1], Catch::Matchers::WithinAbs(20.0, tolerance));
  }
}

TEST_CASE("FVDefaultBoundaries preserves affine stencil derivatives under AD conditioning",
          "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelDefault model;

  SECTION("lower boundary affine ghosts depend on the first two physical cells")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{0.0}}, {{0.0}}, {{1.0}}, {{2.0}}, {{4.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(0.0), Point<1>(0.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};
    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
        {{{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}, {{10}}, {{11}}, {{12}}}};

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 10);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
      CHECK_THAT(derivative(tagged_stencil[1][0]), Catch::Matchers::WithinAbs(2.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[0][0]), Catch::Matchers::WithinAbs(3.0, tolerance));
    }

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 11);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
      CHECK_THAT(derivative(tagged_stencil[1][0]), Catch::Matchers::WithinAbs(-1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[0][0]), Catch::Matchers::WithinAbs(-2.0, tolerance));
    }
  }

  SECTION("upper boundary affine ghosts depend on the last two physical cells")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{1.0}}, {{3.0}}, {{5.0}}, {{0.0}}, {{0.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(7.5), Point<1>(8.5), Point<1>(9.5), Point<1>(0.0), Point<1>(0.0)}};
    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
        {{{20}}, {{21}}, {{22}}, {{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}}};

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 22);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(10.0)));
      CHECK_THAT(derivative(tagged_stencil[3][0]), Catch::Matchers::WithinAbs(2.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[4][0]), Catch::Matchers::WithinAbs(3.0, tolerance));
    }

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 21);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(10.0)));
      CHECK_THAT(derivative(tagged_stencil[3][0]), Catch::Matchers::WithinAbs(-1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[4][0]), Catch::Matchers::WithinAbs(-2.0, tolerance));
    }
  }
}

TEST_CASE("RhoSymmetricLinearExtrapolationBoundaries applies an even lower-boundary stencil",
          "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelRhoSymmetric model;

  SECTION("lower boundary mirrors ghosts evenly and keeps the physical cell free")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{11.0}}, {{12.0}}, {{1.0}}, {{2.0}}, {{4.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(-2.0), Point<1>(-1.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(0.0)));

    CHECK_THAT(x_stencil[0][0], Catch::Matchers::WithinAbs(-1.5, tolerance));
    CHECK_THAT(x_stencil[1][0], Catch::Matchers::WithinAbs(-0.5, tolerance));
    CHECK_THAT(u_stencil[0][0], Catch::Matchers::WithinAbs(2.0, tolerance));
    CHECK_THAT(u_stencil[1][0], Catch::Matchers::WithinAbs(1.0, tolerance));
    CHECK_THAT(u_stencil[2][0], Catch::Matchers::WithinAbs(1.0, tolerance));
  }

  SECTION("upper boundary keeps affine extrapolation")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{1.0}}, {{3.0}}, {{5.0}}, {{17.0}}, {{19.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(7.5), Point<1>(8.5), Point<1>(9.5), Point<1>(20.0), Point<1>(21.0)}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(10.0)));

    CHECK_THAT(x_stencil[3][0], Catch::Matchers::WithinAbs(10.5, tolerance));
    CHECK_THAT(x_stencil[4][0], Catch::Matchers::WithinAbs(11.5, tolerance));
    CHECK_THAT(u_stencil[3][0], Catch::Matchers::WithinAbs(7.0, tolerance));
    CHECK_THAT(u_stencil[4][0], Catch::Matchers::WithinAbs(9.0, tolerance));
  }
}

TEST_CASE("RhoSymmetricLinearExtrapolationBoundaries preserves free physical-cell derivatives",
          "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelRhoSymmetric model;

  SECTION("lower boundary ghosts mirror physical dofs without pinning the zeroth cell")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{0.0}}, {{0.0}}, {{1.0}}, {{2.0}}, {{4.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(0.0), Point<1>(0.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};
    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
        {{{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}, {{10}}, {{11}}, {{12}}}};

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 10);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
      CHECK_THAT(derivative(tagged_stencil[1][0]), Catch::Matchers::WithinAbs(1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[0][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[2][0]), Catch::Matchers::WithinAbs(1.0, tolerance));
    }

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 11);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
      CHECK_THAT(derivative(tagged_stencil[1][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[0][0]), Catch::Matchers::WithinAbs(1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[2][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
    }
  }

  SECTION("upper boundary ghosts keep the affine extrapolation derivatives")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{1.0}}, {{3.0}}, {{5.0}}, {{0.0}}, {{0.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(7.5), Point<1>(8.5), Point<1>(9.5), Point<1>(0.0), Point<1>(0.0)}};
    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
        {{{20}}, {{21}}, {{22}}, {{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}}};

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 22);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(10.0)));
      CHECK_THAT(derivative(tagged_stencil[3][0]), Catch::Matchers::WithinAbs(2.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[4][0]), Catch::Matchers::WithinAbs(3.0, tolerance));
    }

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 21);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(10.0)));
      CHECK_THAT(derivative(tagged_stencil[3][0]), Catch::Matchers::WithinAbs(-1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[4][0]), Catch::Matchers::WithinAbs(-2.0, tolerance));
    }
  }
}

TEST_CASE("Rho symmetric boundary reconstruction matches the mirrored full-domain interior face",
          "[Model][FV][Boundaries][Reconstruction]")
{
  const std::array<double, 3> physical_values{{1.0, 3.0, 5.0}};

  const auto half_domain = rho_lower_boundary_half_domain_reconstruction(physical_values);
  const auto full_domain = rho_full_domain_interior_face_reconstruction(physical_values);

  INFO("half-domain: " << describe_reconstruction(half_domain));
  INFO("full-domain: " << describe_reconstruction(full_domain));

  CHECK_THAT(half_domain.u_plus, Catch::Matchers::WithinAbs(full_domain.u_plus, tolerance));
  CHECK_THAT(half_domain.u_minus, Catch::Matchers::WithinAbs(full_domain.u_minus, tolerance));
  CHECK_THAT(half_domain.grad_u_plus, Catch::Matchers::WithinAbs(full_domain.grad_u_plus, tolerance));
  CHECK_THAT(half_domain.grad_u_minus, Catch::Matchers::WithinAbs(full_domain.grad_u_minus, tolerance));
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK_THAT(half_domain.du_plus[i], Catch::Matchers::WithinAbs(full_domain.du_plus[i], tolerance));
    CHECK_THAT(half_domain.du_minus[i], Catch::Matchers::WithinAbs(full_domain.du_minus[i], tolerance));
    CHECK_THAT(half_domain.dgrad_u_plus[i], Catch::Matchers::WithinAbs(full_domain.dgrad_u_plus[i], tolerance));
    CHECK_THAT(half_domain.dgrad_u_minus[i], Catch::Matchers::WithinAbs(full_domain.dgrad_u_minus[i], tolerance));
  }
}

TEST_CASE("OriginOddLinearExtrapolationBoundaries applies the paper-style two-ghost-cell stencil",
          "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelOriginOddLinearExtrapolation model;

  SECTION("lower boundary sets two odd ghosts and zeroes the origin cell")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{11.0}}, {{12.0}}, {{1.0}}, {{2.0}}, {{4.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(-2.0), Point<1>(-1.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(0.0)));

    CHECK_THAT(x_stencil[0][0], Catch::Matchers::WithinAbs(-1.5, tolerance));
    CHECK_THAT(x_stencil[1][0], Catch::Matchers::WithinAbs(-0.5, tolerance));
    CHECK_THAT(u_stencil[0][0], Catch::Matchers::WithinAbs(-4.0, tolerance));
    CHECK_THAT(u_stencil[1][0], Catch::Matchers::WithinAbs(-2.0, tolerance));
    CHECK_THAT(u_stencil[2][0], Catch::Matchers::WithinAbs(0.0, tolerance));
  }

  SECTION("upper boundary extrapolates two ghosts from the last two physical cells")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{1.0}}, {{3.0}}, {{5.0}}, {{17.0}}, {{19.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(7.5), Point<1>(8.5), Point<1>(9.5), Point<1>(20.0), Point<1>(21.0)}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(10.0)));

    CHECK_THAT(x_stencil[3][0], Catch::Matchers::WithinAbs(10.5, tolerance));
    CHECK_THAT(x_stencil[4][0], Catch::Matchers::WithinAbs(11.5, tolerance));
    CHECK_THAT(u_stencil[3][0], Catch::Matchers::WithinAbs(7.0, tolerance));
    CHECK_THAT(u_stencil[4][0], Catch::Matchers::WithinAbs(9.0, tolerance));
  }
}

TEST_CASE("OriginOddLinearExtrapolationBoundaries applies paper-style lower-axis stencils in 2D",
          "[Model][FV][Boundaries][2d]")
{
  using namespace DiFfRG;
  using namespace def::BoundaryStencilIndex;

  DummyModelOriginOddLinearExtrapolation model;

  // The five named stencil slots form a 1D line along the active boundary normal,
  // embedded in 2D. For lower-x, y is fixed; for lower-y, x is fixed. The lower
  // ghost coordinates start as placeholders and must be overwritten by the helper.
  SECTION("lower-x boundary zeroes the normal component and mirrors the tangential component evenly")
  {
    def::BoundaryStencilValues<2, double, 2> u_stencil{
        {{{11.0, 21.0}}, {{12.0, 22.0}}, {{5.0, 7.0}}, {{2.0, 13.0}}, {{4.0, 17.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}};
    def::BoundaryStencilPoints<2> x_stencil{{Point<2>(8.0, 8.0), Point<2>(9.0, 9.0), Point<2>(0.5, 3.0),
                                             Point<2>(1.5, 3.0), Point<2>(2.5, 3.0), Point<2>(), Point<2>()}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<2>(0.0, 3.0)));

    CHECK_THAT(x_stencil[lower_inner][0], Catch::Matchers::WithinAbs(-0.5, tolerance));
    CHECK_THAT(x_stencil[lower_outer][0], Catch::Matchers::WithinAbs(-1.5, tolerance));
    CHECK_THAT(x_stencil[lower_inner][1], Catch::Matchers::WithinAbs(3.0, tolerance));
    CHECK_THAT(x_stencil[lower_outer][1], Catch::Matchers::WithinAbs(3.0, tolerance));
    CHECK_THAT(u_stencil[physical_cell][0], Catch::Matchers::WithinAbs(0.0, tolerance));
    CHECK_THAT(u_stencil[lower_inner][0], Catch::Matchers::WithinAbs(-2.0, tolerance));
    CHECK_THAT(u_stencil[lower_outer][0], Catch::Matchers::WithinAbs(-4.0, tolerance));
    CHECK_THAT(u_stencil[physical_cell][1], Catch::Matchers::WithinAbs(7.0, tolerance));
    CHECK_THAT(u_stencil[lower_inner][1], Catch::Matchers::WithinAbs(7.0, tolerance));
    CHECK_THAT(u_stencil[lower_outer][1], Catch::Matchers::WithinAbs(13.0, tolerance));
  }

  SECTION("lower-y boundary zeroes the normal component and mirrors the tangential component evenly")
  {
    def::BoundaryStencilValues<2, double, 2> u_stencil{
        {{{11.0, 21.0}}, {{12.0, 22.0}}, {{5.0, 7.0}}, {{2.0, 13.0}}, {{4.0, 17.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}};
    def::BoundaryStencilPoints<2> x_stencil{{Point<2>(8.0, 8.0), Point<2>(9.0, 9.0), Point<2>(3.0, 0.5),
                                             Point<2>(3.0, 1.5), Point<2>(3.0, 2.5), Point<2>(), Point<2>()}};

    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<2>(3.0, 0.0)));

    CHECK_THAT(x_stencil[lower_inner][0], Catch::Matchers::WithinAbs(3.0, tolerance));
    CHECK_THAT(x_stencil[lower_outer][0], Catch::Matchers::WithinAbs(3.0, tolerance));
    CHECK_THAT(x_stencil[lower_inner][1], Catch::Matchers::WithinAbs(-0.5, tolerance));
    CHECK_THAT(x_stencil[lower_outer][1], Catch::Matchers::WithinAbs(-1.5, tolerance));
    CHECK_THAT(u_stencil[physical_cell][0], Catch::Matchers::WithinAbs(5.0, tolerance));
    CHECK_THAT(u_stencil[lower_inner][0], Catch::Matchers::WithinAbs(5.0, tolerance));
    CHECK_THAT(u_stencil[lower_outer][0], Catch::Matchers::WithinAbs(2.0, tolerance));
    CHECK_THAT(u_stencil[physical_cell][1], Catch::Matchers::WithinAbs(0.0, tolerance));
    CHECK_THAT(u_stencil[lower_inner][1], Catch::Matchers::WithinAbs(-13.0, tolerance));
    CHECK_THAT(u_stencil[lower_outer][1], Catch::Matchers::WithinAbs(-17.0, tolerance));
  }
}

TEST_CASE("OriginOddLinearExtrapolationBoundaries keeps affine upper-boundary fallback in 2D",
          "[Model][FV][Boundaries][2d]")
{
  using namespace DiFfRG;
  using namespace def::BoundaryStencilIndex;

  DummyModelOriginOddLinearExtrapolation model;

  // The five named stencil slots form a 1D line along the active boundary normal,
  // embedded in 2D. At an upper-x boundary the upper slots are ghost positions,
  // while the lower slots and physical cell are the interior data.
  def::BoundaryStencilValues<2, double, 2> u_stencil{
      {{{1.0, 3.0}}, {{4.0, 9.0}}, {{7.0, 15.0}}, {{17.0, 18.0}}, {{19.0, 20.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}};
  def::BoundaryStencilPoints<2> x_stencil{{Point<2>(7.5, 2.0), Point<2>(8.5, 2.0), Point<2>(9.5, 2.0),
                                           Point<2>(20.0, 21.0), Point<2>(21.0, 22.0), Point<2>(), Point<2>()}};

  REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<2>(10.0, 2.0)));

  CHECK_THAT(x_stencil[upper_inner][0], Catch::Matchers::WithinAbs(10.5, tolerance));
  CHECK_THAT(x_stencil[upper_outer][0], Catch::Matchers::WithinAbs(11.5, tolerance));
  CHECK_THAT(x_stencil[upper_inner][1], Catch::Matchers::WithinAbs(2.0, tolerance));
  CHECK_THAT(x_stencil[upper_outer][1], Catch::Matchers::WithinAbs(2.0, tolerance));
  CHECK_THAT(u_stencil[upper_inner][0], Catch::Matchers::WithinAbs(10.0, tolerance));
  CHECK_THAT(u_stencil[upper_outer][0], Catch::Matchers::WithinAbs(13.0, tolerance));
  CHECK_THAT(u_stencil[upper_inner][1], Catch::Matchers::WithinAbs(21.0, tolerance));
  CHECK_THAT(u_stencil[upper_outer][1], Catch::Matchers::WithinAbs(27.0, tolerance));
}

	TEST_CASE("OriginOddLinearExtrapolationBoundaries preserves stencil derivatives under AD conditioning",
	          "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelOriginOddLinearExtrapolation model;

  SECTION("lower boundary ghosts depend on interior cells and the zeroed cell has no derivative")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{0.0}}, {{0.0}}, {{1.0}}, {{2.0}}, {{4.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(0.0), Point<1>(0.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};
    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
        {{{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}, {{10}}, {{11}}, {{12}}}};

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 11);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
      CHECK_THAT(derivative(tagged_stencil[1][0]), Catch::Matchers::WithinAbs(-1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[0][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[2][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
    }

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 12);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
      CHECK_THAT(derivative(tagged_stencil[1][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[0][0]), Catch::Matchers::WithinAbs(-1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[2][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
    }

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 10);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
      CHECK_THAT(derivative(tagged_stencil[2][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
    }
  }

  SECTION("upper boundary ghosts keep the affine extrapolation derivatives")
  {
    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{1.0}}, {{3.0}}, {{5.0}}, {{0.0}}, {{0.0}}}};
    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(7.5), Point<1>(8.5), Point<1>(9.5), Point<1>(0.0), Point<1>(0.0)}};
    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
        {{{20}}, {{21}}, {{22}}, {{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}}};

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 22);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(10.0)));
      CHECK_THAT(derivative(tagged_stencil[3][0]), Catch::Matchers::WithinAbs(2.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[4][0]), Catch::Matchers::WithinAbs(3.0, tolerance));
    }

    {
      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 21);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(10.0)));
      CHECK_THAT(derivative(tagged_stencil[3][0]), Catch::Matchers::WithinAbs(-1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[4][0]), Catch::Matchers::WithinAbs(-2.0, tolerance));
    }
	  }
	}

TEST_CASE("OriginOddLinearExtrapolationBoundaries preserves lower-axis 2D derivatives under AD conditioning",
          "[Model][FV][Boundaries][2d]")
{
  using namespace DiFfRG;
  using namespace def::BoundaryStencilIndex;

  DummyModelOriginOddLinearExtrapolation model;

  def::BoundaryStencilValues<2, double, 2> u_stencil{
      {{{11.0, 21.0}}, {{12.0, 22.0}}, {{5.0, 7.0}}, {{2.0, 13.0}}, {{4.0, 17.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}};
  const std::array<std::array<types::global_dof_index, 2>, 7> dof_stencil{
      {{{numbers::invalid_dof_index, numbers::invalid_dof_index}},
       {{numbers::invalid_dof_index, numbers::invalid_dof_index}},
       {{20, 21}},
       {{22, 23}},
       {{24, 25}},
       {{numbers::invalid_dof_index, numbers::invalid_dof_index}},
       {{numbers::invalid_dof_index, numbers::invalid_dof_index}}}};

  const auto check_lower_axis_derivatives = [&](const char *axis_name, const def::BoundaryStencilPoints<2> &x_stencil,
                                                const Point<2> &x_face, const unsigned int normal_component,
                                                const unsigned int tangential_component) {
    INFO(axis_name);

    const auto condition_for_dof = [&](const types::global_dof_index dof_j) {
      auto tagged_stencil = make_tagged_stencil<2>(u_stencil, dof_stencil, dof_j);
      auto conditioned_points = x_stencil;
      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, x_face));
      return tagged_stencil;
    };

    {
      const auto tagged_stencil = condition_for_dof(dof_stencil[upper_inner][normal_component]);
      CHECK_THAT(derivative(tagged_stencil[lower_inner][normal_component]),
                 Catch::Matchers::WithinAbs(-1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_outer][normal_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[physical_cell][normal_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_inner][tangential_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
    }

    {
      const auto tagged_stencil = condition_for_dof(dof_stencil[upper_outer][normal_component]);
      CHECK_THAT(derivative(tagged_stencil[lower_inner][normal_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_outer][normal_component]),
                 Catch::Matchers::WithinAbs(-1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[physical_cell][normal_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_outer][tangential_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
    }

    {
      const auto tagged_stencil = condition_for_dof(dof_stencil[physical_cell][normal_component]);
      CHECK_THAT(derivative(tagged_stencil[physical_cell][normal_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_inner][normal_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_outer][normal_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
    }

    {
      const auto tagged_stencil = condition_for_dof(dof_stencil[physical_cell][tangential_component]);
      CHECK_THAT(derivative(tagged_stencil[physical_cell][tangential_component]),
                 Catch::Matchers::WithinAbs(1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_inner][tangential_component]),
                 Catch::Matchers::WithinAbs(1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_outer][tangential_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_inner][normal_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
    }

    {
      const auto tagged_stencil = condition_for_dof(dof_stencil[upper_inner][tangential_component]);
      CHECK_THAT(derivative(tagged_stencil[physical_cell][tangential_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_inner][tangential_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_outer][tangential_component]),
                 Catch::Matchers::WithinAbs(1.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_outer][normal_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
    }

    {
      const auto tagged_stencil = condition_for_dof(dof_stencil[upper_outer][tangential_component]);
      CHECK_THAT(derivative(tagged_stencil[physical_cell][tangential_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_inner][tangential_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
      CHECK_THAT(derivative(tagged_stencil[lower_outer][tangential_component]),
                 Catch::Matchers::WithinAbs(0.0, tolerance));
    }
  };

  const def::BoundaryStencilPoints<2> lower_x_stencil{{Point<2>(8.0, 8.0), Point<2>(9.0, 9.0),
                                                       Point<2>(0.5, 3.0), Point<2>(1.5, 3.0),
                                                       Point<2>(2.5, 3.0), Point<2>(), Point<2>()}};
  const def::BoundaryStencilPoints<2> lower_y_stencil{{Point<2>(8.0, 8.0), Point<2>(9.0, 9.0),
                                                       Point<2>(3.0, 0.5), Point<2>(3.0, 1.5),
                                                       Point<2>(3.0, 2.5), Point<2>(), Point<2>()}};

  check_lower_axis_derivatives("lower-x", lower_x_stencil, Point<2>(0.0, 3.0), 0, 1);
  check_lower_axis_derivatives("lower-y", lower_y_stencil, Point<2>(3.0, 0.0), 1, 0);
}

	TEST_CASE("OriginShiftedOddLinearExtrapolationBoundaries applies a shifted origin stencil",
	          "[Model][FV][Boundaries]")
	{
	  using namespace DiFfRG;

	  DummyModelOriginShiftedOddLinearExtrapolation model;
	  constexpr double b = DummyModelOriginShiftedOddLinearExtrapolation::origin_value;

	  SECTION("lower boundary reflects around the origin value")
	  {
	    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{11.0}}, {{12.0}}, {{1.0}}, {{2.0}}, {{4.0}}}};
	    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(-2.0), Point<1>(-1.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};

	    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(0.0)));

	    CHECK_THAT(x_stencil[0][0], Catch::Matchers::WithinAbs(-1.5, tolerance));
	    CHECK_THAT(x_stencil[1][0], Catch::Matchers::WithinAbs(-0.5, tolerance));
	    CHECK_THAT(u_stencil[0][0], Catch::Matchers::WithinAbs(2.0 * b - 4.0, tolerance));
	    CHECK_THAT(u_stencil[1][0], Catch::Matchers::WithinAbs(2.0 * b - 2.0, tolerance));
	    CHECK_THAT(u_stencil[2][0], Catch::Matchers::WithinAbs(b, tolerance));
	  }

	  SECTION("upper boundary keeps affine extrapolation")
	  {
	    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{1.0}}, {{3.0}}, {{5.0}}, {{17.0}}, {{19.0}}}};
	    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(7.5), Point<1>(8.5), Point<1>(9.5), Point<1>(20.0), Point<1>(21.0)}};

	    REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<1>(10.0)));

	    CHECK_THAT(x_stencil[3][0], Catch::Matchers::WithinAbs(10.5, tolerance));
	    CHECK_THAT(x_stencil[4][0], Catch::Matchers::WithinAbs(11.5, tolerance));
	    CHECK_THAT(u_stencil[3][0], Catch::Matchers::WithinAbs(7.0, tolerance));
	    CHECK_THAT(u_stencil[4][0], Catch::Matchers::WithinAbs(9.0, tolerance));
	  }
	}

	TEST_CASE("OriginShiftedOddLinearExtrapolationBoundaries preserves stencil derivatives under AD conditioning",
	          "[Model][FV][Boundaries]")
	{
	  using namespace DiFfRG;

	  DummyModelOriginShiftedOddLinearExtrapolation model;

	  SECTION("lower boundary derivatives match shifted odd reflection")
	  {
	    def::BoundaryStencilValues<1, double, 1> u_stencil{{{{0.0}}, {{0.0}}, {{1.0}}, {{2.0}}, {{4.0}}}};
	    def::BoundaryStencilPoints<1> x_stencil{{Point<1>(0.0), Point<1>(0.0), Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};
	    const std::array<std::array<types::global_dof_index, 1>, 5> dof_stencil{
	        {{{numbers::invalid_dof_index}}, {{numbers::invalid_dof_index}}, {{10}}, {{11}}, {{12}}}};

	    {
	      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 11);
	      auto conditioned_points = x_stencil;
	      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
	      CHECK_THAT(derivative(tagged_stencil[1][0]), Catch::Matchers::WithinAbs(-1.0, tolerance));
	      CHECK_THAT(derivative(tagged_stencil[0][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
	      CHECK_THAT(derivative(tagged_stencil[2][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
	    }

	    {
	      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 12);
	      auto conditioned_points = x_stencil;
	      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
	      CHECK_THAT(derivative(tagged_stencil[1][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
	      CHECK_THAT(derivative(tagged_stencil[0][0]), Catch::Matchers::WithinAbs(-1.0, tolerance));
	      CHECK_THAT(derivative(tagged_stencil[2][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
	    }

	    {
	      auto tagged_stencil = make_tagged_stencil<1>(u_stencil, dof_stencil, 10);
	      auto conditioned_points = x_stencil;
	      REQUIRE(model.apply_boundary_stencil(tagged_stencil, conditioned_points, Point<1>(0.0)));
	      CHECK_THAT(derivative(tagged_stencil[2][0]), Catch::Matchers::WithinAbs(0.0, tolerance));
	    }
	  }
	}

	TEST_CASE("Shifted origin boundary reconstruction reduces to zero-origin reconstruction after subtracting shift",
	          "[Model][FV][Boundaries][Reconstruction]")
	{
	  constexpr double b = DummyModelOriginShiftedOddLinearExtrapolation::origin_value;
	  const std::array<double, 3> zero_origin_values{{1.0, 3.0, 5.0}};
	  const std::array<double, 3> shifted_values{{zero_origin_values[0] + b, zero_origin_values[1] + b,
	                                              zero_origin_values[2] + b}};

	  const auto zero_origin = lower_boundary_half_domain_reconstruction(zero_origin_values);
	  const auto shifted = shifted_lower_boundary_half_domain_reconstruction(shifted_values);

	  CHECK_THAT(shifted.u_plus - b, Catch::Matchers::WithinAbs(zero_origin.u_plus, tolerance));
	  CHECK_THAT(shifted.u_minus - b, Catch::Matchers::WithinAbs(zero_origin.u_minus, tolerance));
	  CHECK_THAT(shifted.grad_u_plus, Catch::Matchers::WithinAbs(zero_origin.grad_u_plus, tolerance));
	  CHECK_THAT(shifted.grad_u_minus, Catch::Matchers::WithinAbs(zero_origin.grad_u_minus, tolerance));
	  for (std::size_t i = 0; i < 3; ++i) {
	    CHECK_THAT(shifted.du_plus[i], Catch::Matchers::WithinAbs(zero_origin.du_plus[i], tolerance));
	    CHECK_THAT(shifted.du_minus[i], Catch::Matchers::WithinAbs(zero_origin.du_minus[i], tolerance));
	    CHECK_THAT(shifted.dgrad_u_plus[i], Catch::Matchers::WithinAbs(zero_origin.dgrad_u_plus[i], tolerance));
	    CHECK_THAT(shifted.dgrad_u_minus[i], Catch::Matchers::WithinAbs(zero_origin.dgrad_u_minus[i], tolerance));
	  }
	}

	TEST_CASE("Origin boundary reconstruction differs from the mirrored full-domain interior face",
	          "[Model][FV][Boundaries][Reconstruction]")
{
  SECTION("deterministic non-clipped stencil exposes the dropped origin dof")
  {
    const std::array<double, 3> physical_values{{1.0, 3.0, 5.0}};

    const auto half_domain = lower_boundary_half_domain_reconstruction(physical_values);
    const auto full_domain = origin_full_domain_interior_face_reconstruction(physical_values);

    INFO("half-domain: " << describe_reconstruction(half_domain));
    INFO("full-domain: " << describe_reconstruction(full_domain));

    REQUIRE(differs(half_domain, full_domain));
    CHECK_THAT(half_domain.du_minus[0], Catch::Matchers::WithinAbs(0.0, tolerance));
    CHECK(std::abs(full_domain.du_minus[0]) > mismatch_tolerance);
  }

  SECTION("deterministic clipped stencil shows limiter flattening")
  {
    const std::array<double, 3> physical_values{{1.0, 0.0, 2.0}};

    const auto half_domain = lower_boundary_half_domain_reconstruction(physical_values);
    const auto full_domain = origin_full_domain_interior_face_reconstruction(physical_values);

    INFO("half-domain: " << describe_reconstruction(half_domain));
    INFO("full-domain: " << describe_reconstruction(full_domain));

    CHECK_THAT(full_domain.u_minus, Catch::Matchers::WithinAbs(physical_values[0], tolerance));
    CHECK_THAT(half_domain.u_minus, Catch::Matchers::WithinAbs(0.0, tolerance));
    REQUIRE(differs(half_domain, full_domain));
  }

  SECTION("randomized stable stencils keep showing the mismatch away from limiter kinks")
  {
    std::mt19937 rng(20260508u);
    constexpr std::size_t accepted_trials = 16;
    std::size_t matches = 0;

    for (std::size_t trial = 0; trial < accepted_trials; ++trial) {
      std::array<double, 3> physical_values{};
      do {
        physical_values = draw_random_values(rng);
      } while (!origin_random_trial_is_stable(physical_values));

      const auto half_domain = lower_boundary_half_domain_reconstruction(physical_values);
      const auto full_domain = origin_full_domain_interior_face_reconstruction(physical_values);

      INFO("trial " << trial << ", half-domain: " << describe_reconstruction(half_domain));
      INFO("trial " << trial << ", full-domain: " << describe_reconstruction(full_domain));

      if (!differs(half_domain, full_domain)) ++matches;
      REQUIRE(differs(half_domain, full_domain));
    }

    INFO("origin reconstruction accidental matches in randomized trials: " << matches);
  }
}
