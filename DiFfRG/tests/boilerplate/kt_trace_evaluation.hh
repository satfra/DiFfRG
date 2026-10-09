#pragma once

// DiFfRG
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/assembler/trace_batch.hh>

// standard library
#include <array>
#include <limits>
#include <tuple>
#include <vector>

/**
 * The flux, diffusion flux and their derivatives at single face traces, computed by the same batched routines the KT
 * assembler runs (DiFfRG::FV::KurganovTadmor::internal::flux_value_jacobians, flux_derivatives,
 * diffusion_flux_jacobians), on batches holding just the given traces.
 */
namespace DiFfRG::Testing
{
  namespace kt_detail
  {
    namespace KTI = DiFfRG::FV::KurganovTadmor::internal;

    template <int dim, typename NT, size_t n> using Gradient = KTI::GradientType<dim, NT, n>;
    template <int dim, typename NT, size_t n> using Third = KTI::ThirdDerivativeType<dim, NT, n>;

    /// A batch of the given traces (values, gradients and, for a diffusion batch, 1D third derivatives).
    template <KTI::TraceKind kind, int dim, typename NT, size_t n, typename Extractors, typename Variables>
    auto make_traces(const std::vector<std::array<NT, n>> &u, const std::vector<Gradient<dim, NT, n>> &grad,
                     const std::vector<Third<dim, NT, n>> &third, const dealii::Point<dim> &x,
                     const std::vector<double> &cell_width, const Extractors &extractors, const Variables &variables)
    {
      KTI::TraceBatch<dim, NT, n, Extractors, Variables, kind> batch;
      batch.reinit(u.size());
      for (size_t i = 0; i < u.size(); ++i) {
        for (size_t c = 0; c < n; ++c) {
          batch.value(c, i) = u[i][c];
          for (int d = 0; d < dim; ++d)
            batch.derivative(c, d, i) = grad[i][c][d];
          if constexpr (decltype(batch)::stores_third) batch.third_derivative(c, 0, 0, 0, i) = third[i][c][0][0][0];
        }
        for (int d = 0; d < dim; ++d)
          batch.coordinate(d, i) = x[d];
        batch.width(i) = cell_width[i];
      }
      batch.set_shared(extractors, variables);
      return batch;
    }

    template <typename Model, typename Out, typename Batch>
    concept has_evaluate_batch =
        requires(const Model &m, Out &out, const Batch &batch) { m.evaluate_batch(out, batch); };

    /// The model's evaluate_batch, or, for a bare test model without one, its per-point callbacks.
    template <typename Model, typename Out, typename Batch>
    void evaluate(const Model &model, Out &out, const Batch &batch)
    {
      if constexpr (has_evaluate_batch<Model, Out, Batch>)
        model.evaluate_batch(out, batch);
      else
        DiFfRG::internal::evaluate_per_point(model, out, batch);
    }

    template <typename Model> auto evaluator(const Model &model)
    {
      return [&model](auto &out, const auto &batch) { evaluate(model, out, batch); };
    }
  } // namespace kt_detail

  /// F, dF/du, d2F/du2, dF/dgrad(u) and d2F/(du dgrad(u)) at one trace.
  template <typename Model, typename NT, int dim, size_t n, typename Extractors, typename Variables>
  auto compute_flux_derivatives_ad(const std::array<NT, n> &u, const kt_detail::Gradient<dim, NT, n> &grad_u,
                                   const dealii::Point<dim> &x, const double cell_width, const Extractors &extractors,
                                   const Variables &variables, const Model &model)
  {
    using namespace kt_detail;
    const auto batch =
        make_traces<KTI::TraceKind::flux, dim, NT, n>({u}, {grad_u}, {{}}, x, {cell_width}, extractors, variables);
    KTI::FluxDerivativeWorkspace<std::remove_const_t<decltype(batch)>, n> workspace;
    std::vector<KTI::FluxDerivativeData<NT, dim, n>> result;
    KTI::flux_derivatives<n>(workspace, batch, 0, 1, std::numeric_limits<size_t>::max(), evaluator(model), result);
    return result[0];
  }

  /// F, dF/du and d2F/du2 at one trace with a zero gradient.
  template <typename Model, typename NT, int dim, size_t n, typename Extractors, typename Variables>
  auto compute_flux_jacobian_and_hessian(const std::array<NT, n> &u, const dealii::Point<dim> &x,
                                         const double cell_width, const Extractors &extractors,
                                         const Variables &variables, const Model &model)
  {
    const auto d = compute_flux_derivatives_ad<Model, NT, dim, n>(u, kt_detail::Gradient<dim, NT, n>{}, x, cell_width,
                                                                  extractors, variables, model);
    return std::make_tuple(d.F, d.J, d.H);
  }

  /// The traces' fluxes and the wave speeds of a face (kt_flux_from_traces), the flux jacobians from the batched
  /// first-order AD the assembler's residual uses.
  template <typename WaveSpeedStrategy, typename Model, typename NT, int dim, size_t n, typename Extractors,
            typename Variables>
  auto compute_kt_flux_and_speeds(const std::array<NT, n> &u_plus, const std::array<NT, n> &u_minus,
                                  const kt_detail::Gradient<dim, NT, n> &grad_u_plus,
                                  const kt_detail::Gradient<dim, NT, n> &grad_u_minus, const dealii::Point<dim> &x,
                                  const double cell_width_plus, const double cell_width_minus,
                                  const Extractors &extractors, const Variables &variables, const Model &model)
  {
    using namespace kt_detail;
    const auto batch =
        make_traces<KTI::TraceKind::flux, dim, NT, n>({u_plus, u_minus}, {grad_u_plus, grad_u_minus}, {{}, {}}, x,
                                                      {cell_width_plus, cell_width_minus}, extractors, variables);
    KTI::StackedWorkspace<autodiff::Real<1, NT>, std::remove_const_t<decltype(batch)>, n> workspace;
    std::vector<std::array<dealii::Tensor<1, dim, NT>, n>> F;
    std::vector<std::array<KTI::JacobianMatrix<NT, n>, dim>> J;
    KTI::flux_value_jacobians<n>(workspace, batch, 0, 2, std::numeric_limits<size_t>::max(), evaluator(model), F, J);
    return KTI::kt_flux_from_traces<WaveSpeedStrategy, Model, NT, dim, n>(F[0], F[1], J[0], J[1], model);
  }

  template <typename WaveSpeedStrategy, typename Model, typename NT, int dim, size_t n, typename Extractors,
            typename Variables>
  auto compute_kt_flux_and_speeds(const std::array<NT, n> &u_plus, const std::array<NT, n> &u_minus,
                                  const dealii::Point<dim> &x, const double cell_width_plus,
                                  const double cell_width_minus, const Extractors &extractors,
                                  const Variables &variables, const Model &model)
  {
    return compute_kt_flux_and_speeds<WaveSpeedStrategy, Model, NT, dim, n>(
        u_plus, u_minus, kt_detail::Gradient<dim, NT, n>{}, kt_detail::Gradient<dim, NT, n>{}, x, cell_width_plus,
        cell_width_minus, extractors, variables, model);
  }

  /// The diffusion flux of a face: the average of the two traces' diffusion fluxes.
  template <typename Model, typename NT, int dim, size_t n, typename Extractors, typename Variables>
  auto compute_diffusion_flux(const std::array<NT, n> &u_minus, const std::array<NT, n> &u_plus,
                              const kt_detail::Gradient<dim, NT, n> &grad_u_minus,
                              const kt_detail::Gradient<dim, NT, n> &grad_u_plus,
                              const kt_detail::Third<dim, NT, n> &third_minus,
                              const kt_detail::Third<dim, NT, n> &third_plus, const dealii::Point<dim> &x,
                              const double cell_width_minus, const double cell_width_plus, const Extractors &extractors,
                              const Variables &variables, const Model &model)
  {
    using namespace kt_detail;
    const auto batch = make_traces<KTI::TraceKind::diffusion, dim, NT, n>(
        {u_minus, u_plus}, {grad_u_minus, grad_u_plus}, {third_minus, third_plus}, x,
        {cell_width_minus, cell_width_plus}, extractors, variables);
    BatchOutput<dim, NT, n> out;
    out.reinit(2, Term::diffusion_flux);
    evaluator(model)(out, batch);
    std::array<dealii::Tensor<1, dim, NT>, n> D{};
    for (size_t c = 0; c < n; ++c)
      for (int d = 0; d < dim; ++d)
        D[c][d] = NT(0.5) * (out.diffusion_flux(c, d)[0] + out.diffusion_flux(c, d)[1]);
    return D;
  }

  /// The derivative of the face's diffusion flux with respect to each trace's inputs: [0] the minus, [1] the plus
  /// trace.
  template <typename Model, typename NT, int dim, size_t n, typename Extractors, typename Variables>
  auto compute_diffusion_flux_jacobian(const std::array<NT, n> &u_minus, const std::array<NT, n> &u_plus,
                                       const kt_detail::Gradient<dim, NT, n> &grad_u_minus,
                                       const kt_detail::Gradient<dim, NT, n> &grad_u_plus,
                                       const kt_detail::Third<dim, NT, n> &third_minus,
                                       const kt_detail::Third<dim, NT, n> &third_plus, const dealii::Point<dim> &x,
                                       const double cell_width_minus, const double cell_width_plus,
                                       const Extractors &extractors, const Variables &variables, const Model &model)
  {
    using namespace kt_detail;
    const auto batch = make_traces<KTI::TraceKind::diffusion, dim, NT, n>(
        {u_minus, u_plus}, {grad_u_minus, grad_u_plus}, {third_minus, third_plus}, x,
        {cell_width_minus, cell_width_plus}, extractors, variables);
    KTI::StackedWorkspace<autodiff::Real<1, NT>, std::remove_const_t<decltype(batch)>, n> workspace;
    std::vector<KTI::DiffusionSideJacobian<dim, NT, n>> sides;
    KTI::diffusion_flux_jacobians<n>(workspace, batch, 0, 2, std::numeric_limits<size_t>::max(), evaluator(model),
                                     sides);
    struct {
      std::array<decltype(sides[0].u), 2> u;
      std::array<decltype(sides[0].grad), 2> grad;
      std::array<decltype(sides[0].third_derivatives), 2> third_derivatives;
    } result{{sides[0].u, sides[1].u},
             {sides[0].grad, sides[1].grad},
             {sides[0].third_derivatives, sides[1].third_derivatives}};
    return result;
  }
} // namespace DiFfRG::Testing
