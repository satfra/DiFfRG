#pragma once

// DiFfRG
#include <DiFfRG/discretization/FV/assembler/flux_jacobian_hessian.hh>
#include <DiFfRG/discretization/FV/assembler/flux_ties.hh>
#include <DiFfRG/discretization/FV/reconstructor/types.hh>
#include <DiFfRG/model/batch.hh>

// external libraries
#include <autodiff/forward/real/real.hpp>
#include <deal.II/base/tensor.h>
#include <tbb/tbb.h>

// standard library
#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      namespace internal
      {
        /// Which per-point tuple a TraceBatch hands the model.
        enum class TraceKind {
          /// flux_tie: "fe_functions", "fe_derivatives", "extractors", "variables", "cell_width"
          flux,
          /// diffusion_flux_tie: flux_tie plus "fe_third_derivatives" after the derivatives
          diffusion,
          /// the same names as flux_tie; "fe_derivatives" is the reconstructor's limited slope at the cell point.
          /// There is no "fe_hessians" slot, so a model that reads it without opting in via source_uses_hessians
          /// fails to compile rather than silently receiving zeros.
          source,
          /// source plus "fe_hessians" (the unlimited diagonal curvature) after the derivatives
          source_hessians
        };

        /**
         * @brief The states the KT assembler evaluates the model at, as a batch: the face traces (u^-, u^+ with
         * their reconstructed gradients, and for the diffusion flux the corrected gradients and third derivatives)
         * or the cell quadrature points (limited gradient, and for source_uses_hessians the diagonal curvature).
         *
         * A PointBatch with optional third-derivative columns, `third_derivatives(c, d0, d1, d2)`. Its per-point
         * tuple (tie()) has the names and positions of flux_tie / diffusion_flux_tie (see TraceKind).
         */
        template <int dim_, typename NT, size_t n, typename Extractors, typename Variables, TraceKind kind>
        class TraceBatch
            : public PointBatch<dim_, NT, n, Extractors, Variables, true, kind == TraceKind::source_hessians>
        {
          using Base = PointBatch<dim_, NT, n, Extractors, Variables, true, kind == TraceKind::source_hessians>;

        public:
          static constexpr int dim = dim_;
          static constexpr bool with_third = kind == TraceKind::diffusion;
          /// Third derivatives are only reconstructed in 1D: elsewhere the tuple carries zeros and nothing is stored.
          static constexpr bool stores_third = with_third && dim == 1;
          /// The third derivatives come before the extractors in the diffusion tuple.
          static constexpr uint extractor_index = Base::extractor_index + with_third;
          /// The term this tuple is made for (see DiFfRG::batch_terms).
          static constexpr Term terms = kind == TraceKind::flux        ? Term::flux
                                        : kind == TraceKind::diffusion ? Term::diffusion_flux
                                                                       : Term::source;
          /// Extractors stay double: KT freezes them within a Newton step.
          template <typename NT2> using rebind = TraceBatch<dim, NT2, n, Extractors, Variables, kind>;

          struct State : PointState<dim, NT, n> {
            def::ThirdDerivativeType<dim, NT, n> third_derivatives{};
          };

          void reinit(const size_t n_points)
          {
            Base::reinit(n_points);
            if constexpr (stores_third)
              third.resize(n_points * n_third_columns);
            else if constexpr (with_third)
              third.assign(n_points, NT(0)); // one zero column, the value of every third derivative
          }

          PointSpan<const NT> third_derivatives(const size_t c, const size_t d0, const size_t d1, const size_t d2) const
          {
            static_assert(with_third, "Only the diffusion traces carry third derivatives.");
            if constexpr (!stores_third) return {third.data(), this->size()};
            return {third.data() + third_column(c, d0, d1, d2) * this->size(), this->size()};
          }
          NT &third_derivative(const size_t c, const size_t d0, const size_t d1, const size_t d2, const size_t i)
          {
            static_assert(stores_third, "Third derivatives are only stored in 1D.");
            return third[third_column(c, d0, d1, d2) * this->size() + i];
          }

          void load(const size_t i, State &s) const
          {
            Base::load(i, s);
            if constexpr (stores_third)
              for (size_t c = 0; c < n; ++c)
                for (int d0 = 0; d0 < dim; ++d0)
                  for (int d1 = 0; d1 < dim; ++d1)
                    for (int d2 = 0; d2 < dim; ++d2)
                      s.third_derivatives[c][d0][d1][d2] = third_derivatives(c, d0, d1, d2)[i];
          }

          auto tie(State &s) const
          {
            if constexpr (with_third)
              return diffusion_flux_tie(s.values, s.derivatives, s.third_derivatives, this->extractors(),
                                        this->variables(), s.cell_width);
            else if constexpr (Base::has_hessians)
              return named_tuple<
                  std::tuple<decltype(s.values) &, decltype(s.derivatives) &, decltype(s.hessians) &,
                             const Extractors &, const Variables &, double &>,
                  StringSet<"fe_functions", "fe_derivatives", "fe_hessians", "extractors", "variables", "cell_width">>(
                  std::tie(s.values, s.derivatives, s.hessians, this->extractors(), this->variables(), s.cell_width));
            else
              return flux_tie(s.values, s.derivatives, this->extractors(), this->variables(), s.cell_width);
          }

          template <typename Src> void copy_point(const size_t j, const Src &src, const size_t i)
          {
            Base::copy_point(j, src, i);
            if constexpr (stores_third)
              for (size_t c = 0; c < n; ++c)
                for (int d0 = 0; d0 < dim; ++d0)
                  for (int d1 = 0; d1 < dim; ++d1)
                    for (int d2 = 0; d2 < dim; ++d2)
                      third_derivative(c, d0, d1, d2, j) = src.third_derivatives(c, d0, d1, d2)[i];
          }

        private:
          static constexpr size_t n_third_columns = n * dim * dim * dim;
          static constexpr size_t third_column(const size_t c, const size_t d0, const size_t d1, const size_t d2)
          {
            return ((c * dim + d0) * dim + d1) * dim + d2;
          }
          std::vector<NT> third;
        };

        /// The AD batch and output of stacked_directions, kept by the caller across calls: allocating them anew
        /// for every call costs more than the evaluation when the calls are small (face chunks).
        template <typename ADNumber, typename Batch, size_t n_out> struct StackedWorkspace {
          typename Batch::template rebind<ADNumber> ad;
          BatchOutput<Batch::dim, ADNumber, n_out> out;
        };

        /// The number of points stacked_directions evaluates together.
        inline size_t stacked_chunk_size(const size_t n_points, const size_t n_directions, const size_t max_stacked)
        {
          return std::min(n_points, std::max<size_t>(1, max_stacked / n_directions));
        }

        /// The default chunk_done of stacked_directions: does nothing. A named type rather than
        /// decltype([](size_t, size_t) {}), which CUDA 12's front end cannot deduce through.
        struct NoChunkCallback {
          void operator()(size_t, size_t) const {}
        };

        /**
         * @brief Forward-mode derivatives of a batch evaluation along n_directions input directions at every point,
         * with the directions stacked along the point axis as in DiFfRG::internal::seed_stacked_jacobian.
         *
         * The points are taken in chunks of at most max_stacked / n_directions; block b of the AD batch of a chunk
         * is a copy of its points in which seed(ad_batch, direction, slot) has seeded direction b, so one
         * evaluation of at most max_stacked points covers every direction of the chunk. Small chunks keep the AD
         * batch in cache; a chunk is never smaller than one point, nor a direction group than one direction.
         *
         * Only the points [begin, end) of @p batch are evaluated.
         *
         * @tparam ADNumber the AD type, e.g. autodiff::Real<1, double> or autodiff::Real<2, double>
         * @tparam n_out the number of output components of the evaluation
         * @param evaluate callable(BatchOutput<ADNumber> &out, const ADBatch &batch)
         * @param seed callable(ADBatch &batch, size_t direction, size_t slot)
         * @param read callable(const BatchOutput<ADNumber> &out, size_t direction, size_t point, size_t slot)
         * @param chunk_done callable(size_t first_point, size_t n_points), after every direction of a chunk was read;
         * chunks start at begin + multiples of stacked_chunk_size()
         */
        template <typename ADNumber, size_t n_out, typename Batch, typename Evaluate, typename Seed, typename Read,
                  typename ChunkDone = NoChunkCallback>
        void stacked_directions(StackedWorkspace<ADNumber, Batch, n_out> &workspace, const Batch &batch,
                                const size_t begin, const size_t end, const size_t n_directions,
                                const size_t max_stacked, const Term terms, const Evaluate &evaluate, const Seed &seed,
                                const Read &read, const ChunkDone &chunk_done = {})
        {
          using ADBatch = typename Batch::template rebind<ADNumber>;
          static_assert(std::is_same_v<typename ADBatch::extractors_type, typename Batch::extractors_type>,
                        "stacked_directions shares the batch's extractors with the AD batch, so they must stay frozen "
                        "(as in TraceBatch).");
          const size_t n = end - begin;
          if (n == 0 || n_directions == 0) return;

          auto &ad = workspace.ad;
          auto &out = workspace.out;
          const size_t chunk = stacked_chunk_size(n, n_directions, max_stacked);
          const size_t per_group = std::clamp<size_t>(max_stacked / chunk, 1, n_directions);
          for (size_t p0 = begin; p0 < end; p0 += chunk) {
            const size_t m = std::min(chunk, end - p0);
            for (size_t g0 = 0; g0 < n_directions; g0 += per_group) {
              const size_t G = std::min(per_group, n_directions - g0);
              ad.reinit(G * m);
              ad.set_shared(batch.extractors(), batch.variables());
              out.reinit(G * m, terms);
              tbb::parallel_for(tbb::blocked_range<size_t>(0, m), [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i != r.end(); ++i)
                  for (size_t b = 0; b < G; ++b) {
                    ad.copy_point(b * m + i, batch, p0 + i);
                    seed(ad, g0 + b, b * m + i);
                  }
              });
              evaluate(out, std::as_const(ad));
              tbb::parallel_for(tbb::blocked_range<size_t>(0, m), [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i != r.end(); ++i)
                  for (size_t b = 0; b < G; ++b)
                    read(std::as_const(out), g0 + b, p0 + i, b * m + i);
              });
            }
            chunk_done(p0, m);
          }
        }

        /// The flux F and its value jacobian dF/du at the traces [begin, end) of a flux trace batch, into F[i - begin]
        /// and J[i - begin]: first-order forward AD along every value, the directions stacked (stacked_directions).
        template <size_t n, typename Batch, typename Evaluate>
        void
        flux_value_jacobians(StackedWorkspace<autodiff::Real<1, typename Batch::number_type>, Batch, n> &workspace,
                             const Batch &traces, const size_t begin, const size_t end, const size_t max_stacked,
                             const Evaluate &evaluate,
                             std::vector<std::array<dealii::Tensor<1, Batch::dim, typename Batch::number_type>, n>> &F,
                             std::vector<std::array<JacobianMatrix<typename Batch::number_type, n>, Batch::dim>> &J)
        {
          using autodiff::detail::derivative;
          using NT = typename Batch::number_type;
          F.resize(end - begin);
          J.resize(end - begin);
          stacked_directions<autodiff::Real<1, NT>, n>(
              workspace, traces, begin, end, n, max_stacked, Term::flux, evaluate,
              [](auto &batch, const size_t c, const size_t j) { autodiff::detail::seed<1>(batch.value(c, j), NT(1)); },
              [&](const auto &out, const size_t c_in, const size_t i, const size_t j) {
                for (uint c = 0; c < n; ++c)
                  for (int d = 0; d < Batch::dim; ++d) {
                    J[i - begin][d][c][c_in] = derivative<1>(out.flux(c, d)[j]);
                    if (c_in == 0) F[i - begin][c][d] = out.flux(c, d)[j].val();
                  }
              });
        }

        /// The number of directions flux_derivatives() differentiates along per trace: every value, every pair of
        /// values, every gradient entry, and every value with every gradient entry.
        template <int dim, size_t n>
        inline constexpr size_t n_flux_derivative_directions = n + n * (n - 1) / 2 + n * dim + n * n * dim;
        /// The number of directions diffusion_flux_jacobians() differentiates along per trace: every value, every
        /// gradient entry and, in 1D, every third derivative.
        template <int dim, size_t n>
        inline constexpr size_t n_diffusion_derivative_directions = n + n * dim + (dim == 1 ? n : 0);

        /// The buffers of flux_derivatives(), kept by the caller across calls.
        template <typename Batch, size_t n> struct FluxDerivativeWorkspace {
          StackedWorkspace<autodiff::Real<2, typename Batch::number_type>, Batch, n> stacked;
          /// First and second derivative of every flux entry along every direction, for the points of one chunk.
          std::vector<typename Batch::number_type> first, second;
        };

        /**
         * @brief The flux and its derivatives at the traces [begin, end) of a flux trace batch, into result[i - begin]:
         * F, J = dF/du, H = d2F/du2, grad_J = dF/dgrad(u) and mixed_H = d2F/(du dgrad(u)), by second-order forward AD.
         * Off-diagonal second derivatives come from polarisation: the second derivative along u_j + u_c (or u_j +
         * grad(u)_c,d) minus those along each alone. All directions are stacked (stacked_directions).
         */
        template <size_t n, typename Batch, typename Evaluate>
        void flux_derivatives(FluxDerivativeWorkspace<Batch, n> &workspace, const Batch &traces, const size_t begin,
                              const size_t end, const size_t max_stacked, const Evaluate &evaluate,
                              std::vector<FluxDerivativeData<typename Batch::number_type, Batch::dim, n>> &result)
        {
          using autodiff::detail::derivative;
          using autodiff::detail::seed;
          using NT = typename Batch::number_type;
          constexpr int dim = Batch::dim;

          // Directions: u_j (diagonal), u_j + u_c (j < c), grad_{c,d}, u_j + grad_{c,d}.
          struct Direction {
            int u1 = -1, u2 = -1, grad_c = -1, grad_d = -1;
          };
          std::vector<Direction> directions;
          for (uint j = 0; j < n; ++j)
            directions.push_back({int(j), -1, -1, -1});
          for (uint j = 0; j < n; ++j)
            for (uint c = j + 1; c < n; ++c)
              directions.push_back({int(j), int(c), -1, -1});
          for (uint c = 0; c < n; ++c)
            for (int d = 0; d < dim; ++d)
              directions.push_back({-1, -1, int(c), d});
          for (uint j = 0; j < n; ++j)
            for (uint c = 0; c < n; ++c)
              for (int d = 0; d < dim; ++d)
                directions.push_back({int(j), -1, int(c), d});

          const size_t per_point = n * dim;
          const size_t chunk = stacked_chunk_size(end - begin, directions.size(), max_stacked);
          auto &first = workspace.first, &second = workspace.second;
          first.resize(directions.size() * chunk * per_point);
          second.resize(directions.size() * chunk * per_point);
          result.resize(end - begin);
          stacked_directions<autodiff::Real<2, NT>, n>(
              workspace.stacked, traces, begin, end, directions.size(), max_stacked, Term::flux, evaluate,
              [&](auto &batch, const size_t k, const size_t j) {
                const auto &dir = directions[k];
                if (dir.u1 >= 0) seed<1>(batch.value(dir.u1, j), NT(1));
                if (dir.u2 >= 0) seed<1>(batch.value(dir.u2, j), NT(1));
                if (dir.grad_c >= 0) seed<1>(batch.derivative(dir.grad_c, dir.grad_d, j), NT(1));
              },
              [&](const auto &out, const size_t k, const size_t i, const size_t j) {
                for (uint c = 0; c < n; ++c)
                  for (int d = 0; d < dim; ++d) {
                    const size_t at = (k * chunk + (i - begin) % chunk) * per_point + c * dim + d;
                    first[at] = derivative<1>(out.flux(c, d)[j]);
                    second[at] = derivative<2>(out.flux(c, d)[j]);
                    if (k == 0) result[i - begin].F[c][d] = out.flux(c, d)[j].val();
                  }
              },
              // Combine a chunk as soon as all its directions are in.
              [&](const size_t p0, const size_t m) {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, m), [&](const tbb::blocked_range<size_t> &r) {
                  for (size_t l = r.begin(); l != r.end(); ++l) {
                    auto &res = result[p0 + l - begin];
                    FluxGradientJacobian<NT, dim, n> grad_diagonal_H{};
                    const auto d1 = [&](const size_t k, const uint c, const int d) {
                      return first[(k * chunk + l) * per_point + c * dim + d];
                    };
                    const auto d2 = [&](const size_t k, const uint c, const int d) {
                      return second[(k * chunk + l) * per_point + c * dim + d];
                    };
                    size_t k = 0;
                    for (uint j = 0; j < n; ++j, ++k)
                      for (uint c = 0; c < n; ++c)
                        for (int d = 0; d < dim; ++d) {
                          res.J[d][c][j] = d1(k, c, d);
                          res.H[d][c][j][j] = d2(k, c, d);
                        }
                    for (uint j = 0; j < n; ++j)
                      for (uint jc = j + 1; jc < n; ++jc, ++k)
                        for (uint c = 0; c < n; ++c)
                          for (int d = 0; d < dim; ++d)
                            res.H[d][c][j][jc] = res.H[d][c][jc][j] =
                                (d2(k, c, d) - res.H[d][c][j][j] - res.H[d][c][jc][jc]) / NT(2);
                    for (uint gc = 0; gc < n; ++gc)
                      for (int d_in = 0; d_in < dim; ++d_in, ++k)
                        for (uint c = 0; c < n; ++c)
                          for (int d = 0; d < dim; ++d) {
                            res.grad_J[c][gc][d][d_in] = d1(k, c, d);
                            grad_diagonal_H[c][gc][d][d_in] = d2(k, c, d);
                          }
                    for (uint j = 0; j < n; ++j)
                      for (uint gc = 0; gc < n; ++gc)
                        for (int d_in = 0; d_in < dim; ++d_in, ++k)
                          for (uint c = 0; c < n; ++c)
                            for (int d = 0; d < dim; ++d)
                              res.mixed_H[d_in][d][c][j][gc] =
                                  (d2(k, c, d) - res.H[d][c][j][j] - grad_diagonal_H[c][gc][d][d_in]) / NT(2);
                  }
                });
              });
        }

        /**
         * @brief Half the derivative of the diffusion flux at the traces [begin, end) of a diffusion trace batch with
         * respect to every value, gradient and (1D) third-derivative entry, into result[i - begin]; first-order forward
         * AD with the directions stacked. Third derivatives are only reconstructed in 1D; elsewhere they are zero and
         * nothing depends on them.
         */
        template <size_t n, typename Batch, typename Evaluate>
        void
        diffusion_flux_jacobians(StackedWorkspace<autodiff::Real<1, typename Batch::number_type>, Batch, n> &workspace,
                                 const Batch &traces, const size_t begin, const size_t end, const size_t max_stacked,
                                 const Evaluate &evaluate,
                                 std::vector<DiffusionSideJacobian<Batch::dim, typename Batch::number_type, n>> &result)
        {
          using autodiff::detail::derivative;
          using autodiff::detail::seed;
          using NT = typename Batch::number_type;
          constexpr int dim = Batch::dim;
          constexpr uint n_dirs_value = n, n_dirs_grad = n * dim;
          result.assign(end - begin, {});
          stacked_directions<autodiff::Real<1, NT>, n>(
              workspace, traces, begin, end, n_diffusion_derivative_directions<dim, n>, max_stacked,
              Term::diffusion_flux, evaluate,
              [&](auto &batch, size_t k, const size_t j) {
                if (k < n_dirs_value) return seed<1>(batch.value(k, j), NT(1));
                k -= n_dirs_value;
                if (k < n_dirs_grad) return seed<1>(batch.derivative(k / dim, k % dim, j), NT(1));
                if constexpr (dim == 1) seed<1>(batch.third_derivative(k - n_dirs_grad, 0, 0, 0, j), NT(1));
              },
              [&](const auto &out, size_t k, const size_t i, const size_t j) {
                auto &res = result[i - begin];
                for (uint c = 0; c < n; ++c)
                  for (int d = 0; d < dim; ++d) {
                    const NT half = NT(0.5) * derivative<1>(out.diffusion_flux(c, d)[j]);
                    if (k < n_dirs_value) {
                      res.u(c, k)[d] = half;
                      continue;
                    }
                    const size_t kg = k - n_dirs_value;
                    if (kg < n_dirs_grad) {
                      res.grad(c, kg / dim)[d][kg % dim] = half;
                      continue;
                    }
                    res.third_derivatives(c, kg - n_dirs_grad)[d][0][0][0] = half; // 1D only
                  }
              });
        }
      } // namespace internal
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
