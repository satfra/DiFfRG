#pragma once

// DiFfRG
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>

#include <boilerplate/models.hh>

namespace DiFfRG
{
  namespace Testing
  {
    using namespace dealii;

    template <typename NumberType, std::size_t n_components, typename SolutionEvaluator>
    void fill_face_ghost_solution_boundary_stencil(def::BoundaryStencilValues<1, NumberType, n_components> &u_stencil,
                                                   def::BoundaryStencilPoints<1> &x_stencil,
                                                   const Point<1> &x_face, SolutionEvaluator &&solution)
    {
      using Reconstructor = def::TVDReconstructor<1, def::MinModLimiter, NumberType>;
      using namespace def::BoundaryStencilIndex;

      const bool lower_boundary = x_face[0] <= x_stencil[physical_cell][0];
      const auto face_value = solution(x_face);

      if (lower_boundary) {
        x_stencil[lower_inner] = x_face;
        for (size_t c = 0; c < n_components; ++c)
          u_stencil[lower_inner][c] = face_value[c];

        const auto interior_gradient = Reconstructor::template compute_gradient<n_components>(
            x_stencil[physical_cell], u_stencil[physical_cell], {x_stencil[lower_inner], x_stencil[upper_inner]},
            {u_stencil[lower_inner], u_stencil[upper_inner]});

        x_stencil[lower_outer][0] = 2.0 * x_stencil[lower_inner][0] - x_stencil[physical_cell][0];
        for (size_t c = 0; c < n_components; ++c)
          u_stencil[lower_outer][c] =
              u_stencil[lower_inner][c] + interior_gradient[c][0] * (x_stencil[lower_outer][0] - x_stencil[lower_inner][0]);
        return;
      }

      x_stencil[upper_inner] = x_face;
      for (size_t c = 0; c < n_components; ++c)
        u_stencil[upper_inner][c] = face_value[c];

      const auto interior_gradient = Reconstructor::template compute_gradient<n_components>(
          x_stencil[physical_cell], u_stencil[physical_cell], {x_stencil[lower_inner], x_stencil[upper_inner]},
          {u_stencil[lower_inner], u_stencil[upper_inner]});

      x_stencil[upper_outer][0] = 2.0 * x_stencil[upper_inner][0] - x_stencil[physical_cell][0];
      for (size_t c = 0; c < n_components; ++c)
        u_stencil[upper_outer][c] =
            u_stencil[upper_inner][c] + interior_gradient[c][0] * (x_stencil[upper_outer][0] - x_stencil[upper_inner][0]);
    }

    template <uint dim>
    class ModelBurgersKT
        : public def::AbstractModel<ModelBurgersKT<dim>, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
          public def::Time,
          public def::LLFFlux<ModelBurgersKT<dim>>,
          public def::FlowBoundaries<ModelBurgersKT<dim>>,
          public def::FVDefaultBoundaries<ModelBurgersKT<dim>>,
          public def::AD<ModelBurgersKT<dim>>
    {
    protected:
      const PhysicalParameters prm;

    public:
      ModelBurgersKT(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        // The return type is spelled out because nvcc's cudafe strips the braces from a
        // single-element `return {...}` in a class template member, leaving g++ with a bare double.
        return std::array<double, 1>{(prm.initial_x0[0] + prm.initial_x1[0] * pos[0]) /
                                     (prm.initial_x1[0] * t + 1.)};
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
      }

      template <int mdim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NumberType, n_components> &u_stencil,
                                  def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
      {
        static_assert(mdim == 1, "KT boundary stencils in boilerplate models are currently one-dimensional.");
        fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                  [this](const Point<1> &pos) { return solution(pos); });
        return true;
      }
    };

    class ModelBurgers2DKT
        : public def::AbstractModel<ModelBurgers2DKT, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
          public def::Time,
          public def::LLFFlux<ModelBurgers2DKT>,
          public def::FlowBoundaries<ModelBurgers2DKT>,
          public def::AD<ModelBurgers2DKT>
    {
    protected:
      const PhysicalParameters prm;

    public:
      static constexpr uint dim = 2;
      ModelBurgers2DKT(PhysicalParameters prm) : prm(prm) {}

      double denominator() const { return 1.0 + (prm.initial_x1[0] + prm.initial_x2[0]) * t; }

      template <typename Vector> void initial_condition(const Point<2> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0] + prm.initial_x2[0] * pos[1];
      }

      std::array<double, 1> solution(const Point<2> &pos) const
      {
        return {(prm.initial_x0[0] + prm.initial_x1[0] * pos[0] + prm.initial_x2[0] * pos[1]) / denominator()};
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        const auto flux = 0.5 * powr<2>(fe_functions[0]);
        F_i[0][0] = flux;
        F_i[0][1] = flux;
      }

      template <int mdim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NumberType, n_components> &u_stencil,
                                  def::BoundaryStencilPoints<mdim> &x_stencil,
                                  [[maybe_unused]] const Point<mdim> &x_face) const
      {
        static_assert(mdim == 2, "ModelBurgers2DKT boundary stencils are two-dimensional.");
        for (size_t i = 0; i < x_stencil.size(); ++i)
          u_stencil[i][0] = (NumberType(prm.initial_x0[0]) + NumberType(prm.initial_x1[0]) * x_stencil[i][0] +
                             NumberType(prm.initial_x2[0]) * x_stencil[i][1]) /
                            NumberType(denominator());
        return true;
      }
    };

    template <uint dim>
    class ModelBurgersTravelingWaveKT
        : public def::AbstractModel<ModelBurgersTravelingWaveKT<dim>,
                                    ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
          public def::Time,
          public def::LLFFlux<ModelBurgersTravelingWaveKT<dim>>,
          public def::FlowBoundaries<ModelBurgersTravelingWaveKT<dim>>,
          public def::FVDefaultBoundaries<ModelBurgersTravelingWaveKT<dim>>,
          public def::AD<ModelBurgersTravelingWaveKT<dim>>
    {
    protected:
      static constexpr double nu = 10.0;
      static constexpr double f_plus = 2.0;
      static constexpr double f_minus = 0.0;
      static constexpr double c = (f_plus + f_minus) / 2.0;

    public:
      ModelBurgersTravelingWaveKT(PhysicalParameters) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = f_plus / (1.0 + std::exp(pos[0] / nu));
      }

      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        // Explicit return type: see the note in ModelBurgersKT::solution.
        return std::array<double, 1>{f_plus / (1.0 + std::exp((pos[0] - c * t) / nu))};
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
      }

      template <typename NT, typename Solution>
      void diffusion_flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_derivatives = get<1>(sol);
        // KT sums (H + D)·n: the diffusion flux is the physical conservation-law viscous
        // flux -ν·∂u (decreasing in the gradient ⇒ forward diffusion).
        F_i[0] = -nu * fe_derivatives[0];
      }

      template <int mdim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NumberType, n_components> &u_stencil,
                                  def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
      {
        static_assert(mdim == 1, "KT boundary stencils in boilerplate models are currently one-dimensional.");
        fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                  [this](const Point<1> &pos) { return solution(pos); });
        return true;
      }
    };

    template <uint dim>
    class ModelTwoComponentBurgersKT
        : public def::AbstractModel<ModelTwoComponentBurgersKT<dim>,
                                    ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>>>,
          public def::Time,
          public def::LLFFlux<ModelTwoComponentBurgersKT<dim>>,
          public def::FlowBoundaries<ModelTwoComponentBurgersKT<dim>>,
          public def::FVDefaultBoundaries<ModelTwoComponentBurgersKT<dim>>,
          public def::AD<ModelTwoComponentBurgersKT<dim>>
    {
    protected:
      const PhysicalParameters prm;
      static constexpr double nu = 10.0;
      static constexpr double f_plus = 2.0;
      static constexpr double f_minus = 0.0;
      static constexpr double c = (f_plus + f_minus) / 2.0;

    public:
      ModelTwoComponentBurgersKT(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = f_plus / (1.0 + std::exp(pos[0] / nu));
        values[1] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      std::array<double, 2> solution(const Point<dim> &pos) const
      {
        const double u_sol = f_plus / (1.0 + std::exp((pos[0] - c * t) / nu));
        const double v_sol = (prm.initial_x0[0] + prm.initial_x1[0] * pos[0]) / (prm.initial_x1[0] * t + 1.);
        return {u_sol, v_sol};
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, dim, NT>, 2> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
        F_i[1][0] = 0.5 * powr<2>(fe_functions[1]);
      }

      template <typename NT, typename Solution>
      void diffusion_flux(std::array<Tensor<1, dim, NT>, 2> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_derivatives = get<1>(sol);
        // KT sums (H + D)·n: the diffusion flux is the physical conservation-law viscous
        // flux -ν·∂u (decreasing in the gradient ⇒ forward diffusion).
        F_i[0] = -nu * fe_derivatives[0];
        F_i[1] = 0.0;
      }

      template <int mdim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NumberType, n_components> &u_stencil,
                                  def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
      {
        static_assert(mdim == 1, "KT boundary stencils in boilerplate models are currently one-dimensional.");
        fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                  [this](const Point<1> &pos) { return solution(pos); });
        return true;
      }
    };
  } // namespace Testing
} // namespace DiFfRG
