#pragma once

// standard library
#include <array>

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/model/batch.hh>

namespace DiFfRG
{
  namespace def
  {
    using namespace dealii;

    template <typename Model> class LLFFlux
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }

    public:
      template <int dim, typename NumberType, typename Solutions_s, typename Solutions_n, typename M = Model>
      void numflux(std::array<Tensor<1, dim, NumberType>, M::Components::count_fe_functions(0)> &NF,
                   const Tensor<1, dim> &normal, const Point<dim> &p, const Solutions_s &sol_s,
                   const Solutions_n &sol_n) const
      {
        using std::max, std::abs;
        using namespace autodiff;
        static_assert(std::is_same<M, Model>::value, "Internal error: template parameter M must be the same as Model. "
                                                     "Do not explicitly specify the M template parameter.");
        using Components = typename M::Components;

        std::array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> F_s{};
        std::array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> F_n{};
        asImp().flux(F_s, p, sol_s);
        asImp().flux(F_n, p, sol_n);

        const auto &u_s = get<0>(sol_s);
        const auto &u_n = get<0>(sol_n);

        // A lengthy calculation for the diffusion
        // we use FD here, as nested AD calculations would be quite the hassle
        auto du_s = vector_to_array<Components::count_fe_functions(0), NumberType>(u_s);
        auto du_n = vector_to_array<Components::count_fe_functions(0), NumberType>(u_n);
        std::array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> dflux_s{};
        std::array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> dflux_n{};
        for (uint i = 0; i < Model::Components::count_fe_functions(0); ++i) {
          auto du = 1e-5 * (0.5 * (abs(u_s[i]) + abs(u_n[i])) + 1e-9);
          du_s[i] += du;
          du_n[i] += du;
          asImp().flux(dflux_s, p, Solutions_s::as(std::tuple_cat(std::tie(du_s), tuple_tail(sol_s))));
          asImp().flux(dflux_n, p, Solutions_n::as(std::tuple_cat(std::tie(du_n), tuple_tail(sol_n))));
          du_s[i] = u_s[i];
          du_n[i] = u_n[i];

          const auto alpha = max(abs(dot<dim, NumberType>(dflux_s[i], normal) - dot<dim, NumberType>(F_s[i], normal)),
                                 abs(dot<dim, NumberType>(dflux_n[i], normal) - dot<dim, NumberType>(F_n[i], normal))) /
                             du;

          for (uint d = 0; d < dim; ++d)
            NF[i][d] = 0.5 * (F_s[i][d] + F_n[i][d]) - 0.5 * alpha * (u_n[i] - u_s[i]);
        }
      }

      /**
       * @brief numflux at all points of an interior-face batch: the same formula, with every flux it needs
       * (both traces, and each trace with one component perturbed for the finite-difference wave speed)
       * evaluated by ONE evaluate_batch call over 2 + 2 n_fe stacked copies of the traces.
       */
      template <typename Out, typename Normals, typename Batch>
      void numflux_batch(Out &out, const Normals &normals, const Batch &batch_s, const Batch &batch_n) const
      {
        using std::max, std::abs;
        using namespace autodiff;
        using NT = typename Batch::number_type;
        constexpr int dim = Batch::dim;
        // The FE functions; under LDG the batch also holds the LDG levels after them.
        constexpr size_t n_fe = Out::n_components;
        // Block 0: trace s, block 1: trace n, block 2 + 2c (3 + 2c): trace s (n) with component c perturbed.
        constexpr size_t n_blocks = 2 + 2 * n_fe;
        const size_t n = batch_s.size();

        // Kept across calls (one set per number type and calling thread): allocating them anew costs page faults on
        // every call, and the jacobian calls this once per seed group. The parallel loops below must see the calling
        // thread's buffers, hence the references: inside a lambda run by another thread, the name of a thread_local
        // refers to that thread's instance.
        static thread_local Batch stacked_buffer;
        static thread_local BatchOutput<dim, NT, n_fe> F_buffer;
        static thread_local std::vector<NT> du_buffer;
        auto &stacked = stacked_buffer;
        auto &F = F_buffer;
        auto &du = du_buffer;
        stacked.reinit(n_blocks * n);
        stacked.set_shared(batch_s.extractors(), batch_s.variables());
        du.resize(n_fe * n);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            for (size_t b = 0; b < n_blocks; ++b) {
              stacked.copy_point(b * n + i, b % 2 == 0 ? batch_s : batch_n, i);
              if (b < 2) continue;
              const size_t c = (b - 2) / 2;
              const NT u_s = batch_s.values(c)[i], u_n = batch_n.values(c)[i];
              du[c * n + i] = 1e-5 * (0.5 * (abs(u_s) + abs(u_n)) + 1e-9);
              stacked.value(c, b * n + i) += du[c * n + i];
            }
        });

        F.reinit(n_blocks * n, Term::flux);
        asImp().evaluate_batch(F, stacked);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            for (size_t c = 0; c < n_fe; ++c) {
              const auto at = [&](const size_t b, const int d) { return F.flux(c, d)[b * n + i]; };
              NT f_s = 0., f_n = 0., df_s = 0., df_n = 0.;
              for (int d = 0; d < dim; ++d) {
                f_s += at(0, d) * normals[i][d];
                f_n += at(1, d) * normals[i][d];
                df_s += at(2 + 2 * c, d) * normals[i][d];
                df_n += at(3 + 2 * c, d) * normals[i][d];
              }
              const NT alpha = max(abs(df_s - f_s), abs(df_n - f_n)) / du[c * n + i];
              const NT jump = batch_n.values(c)[i] - batch_s.values(c)[i];
              for (int d = 0; d < dim; ++d)
                out.flux(c, d)[i] = 0.5 * (at(0, d) + at(1, d)) - 0.5 * alpha * jump;
            }
        });
        stacked.clear_shared();
      }
    };

    constexpr uint from_right = 0;
    constexpr uint from_left = 1;

    template <typename... T> struct UpDownFlux {
      template <uint i> using value = typename std::tuple_element<i, std::tuple<T...>>::type;
    };
    template <int... n> struct FlowDirections {
      static constexpr std::array<int, sizeof...(n)> value{{n...}};
      static constexpr int size = sizeof...(n);
    };
    template <int... n> struct UpDown {
      static constexpr std::array<int, sizeof...(n)> value{{n...}};
      static constexpr int size = sizeof...(n);
    };
    template <typename Model, typename... Collections> class LDGUpDownFluxes
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      template <int i> using C = typename std::tuple_element<i, std::tuple<Collections...>>::type;

    public:
      template <uint dependent, int dim, typename NumberType, typename Solutions_s, typename Solutions_n,
                typename M = Model>
      void ldg_numflux(std::array<Tensor<1, dim, NumberType>, M::Components::count_fe_functions(dependent)> &NF,
                       const Tensor<1, dim> &normal, const Point<dim> &p, const Solutions_s &u_s,
                       const Solutions_n &u_n) const
      {
        static_assert(std::is_same<M, Model>::value, "Internal error: template parameter M must be the same as Model. "
                                                     "Do not explicitly specify the M template parameter.");
        static_assert(dependent >= 1, "ldg_numflux requires dependent >= 1 (use numflux for dependent == 0).");

        using Dirs = typename C<dependent - 1>::template value<0>;
        using UD = typename C<dependent - 1>::template value<1>;
        static_assert(
            Dirs::size == UD::size && UD::size >= M::Components::count_fe_functions(dependent),
            "LDG numflux: FlowDirections::size and UpDown::size must both be >= count_fe_functions(dependent).");
        using Components = typename M::Components;

        Tensor<1, dim> t;
        for (uint i = 0; i < dim; ++i)
          t[i] = -1.;

        std::array<std::array<Tensor<1, dim, NumberType>, Components::count_fe_functions(dependent)>, 2> F;
        // normals are facing outwards! Therefore, the first case is the one where the normal points to the left
        // (smaller field values), the second case is the one where the normal points to the right (larger field
        // values).
        if (scalar_product(t, normal) >= 0) {
          // F[0] takes the flux from the right (inside the cell), F[1] takes the flux from the left (the other cell)
          asImp().template ldg_flux<dependent>(F[0], p, u_s);
          asImp().template ldg_flux<dependent>(F[1], p, u_n);
        } else {
          // F[0] takes the flux from the right (the other cell), F[1] takes the flux from the left (inside the cell)
          asImp().template ldg_flux<dependent>(F[0], p, u_n);
          asImp().template ldg_flux<dependent>(F[1], p, u_s);
        }

        for (uint i = 0; i < Components::count_fe_functions(dependent); ++i)
          NF[i][Dirs::value[i]] = F[UD::value[i]][i][Dirs::value[i]];
      }

      /// The batched ldg_numflux: ldg_evaluate_batch on both traces, then the same upwind selection per point.
      template <uint dependent, typename Out, typename Normals, typename Batch>
      void ldg_numflux_batch(Out &out, const Normals &normals, const Batch &batch_s, const Batch &batch_n) const
      {
        static_assert(dependent >= 1, "ldg_numflux requires dependent >= 1 (use numflux for dependent == 0).");
        using Dirs = typename C<dependent - 1>::template value<0>;
        using UD = typename C<dependent - 1>::template value<1>;
        constexpr int dim = Batch::dim;
        constexpr size_t n_out = Out::n_components;
        static_assert(
            Dirs::size == UD::size && UD::size >= n_out,
            "LDG numflux: FlowDirections::size and UpDown::size must both be >= count_fe_functions(dependent).");

        const size_t n = batch_s.size();
        // Kept across calls, and referenced from the parallel loop, as in LLFFlux::numflux_batch.
        static thread_local BatchOutput<dim, typename Batch::number_type, n_out> F_s_buffer, F_n_buffer;
        auto &F_s = F_s_buffer;
        auto &F_n = F_n_buffer;
        F_s.reinit(n, Term::flux);
        F_n.reinit(n, Term::flux);
        asImp().template ldg_evaluate_batch<dependent>(F_s, batch_s);
        asImp().template ldg_evaluate_batch<dependent>(F_n, batch_n);

        Tensor<1, dim> t;
        for (int d = 0; d < dim; ++d)
          t[d] = -1.;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i) {
            // As in ldg_numflux: F[0] is the flux from the right, F[1] the one from the left.
            const bool normal_points_left = scalar_product(t, normals[i]) >= 0;
            const std::array<const BatchOutput<dim, typename Batch::number_type, n_out> *, 2> F{
                {normal_points_left ? &F_s : &F_n, normal_points_left ? &F_n : &F_s}};
            for (uint c = 0; c < n_out; ++c)
              out.flux(c, Dirs::value[c])[i] = F[UD::value[c]]->flux(c, Dirs::value[c])[i];
          }
        });
      }
    };

    template <typename Model> class NoNumFlux
    {
    public:
      template <int dim, typename NumberType, typename Solutions_s, typename Solutions_n, typename M = Model>
      void numflux(std::array<Tensor<1, dim, NumberType>, M::Components::count_fe_functions(0)> &,
                   const Tensor<1, dim> &, const Point<dim> &, const Solutions_s &, const Solutions_n &) const
      {
        static_assert(std::is_same<M, Model>::value, "Internal error: template parameter M must be the same as Model. "
                                                     "Do not explicitly specify the M template parameter.");
      }

      /// The batched numflux: nothing, the output is zero on entry.
      template <typename Out, typename Normals, typename Batch>
      void numflux_batch(Out &, const Normals &, const Batch &, const Batch &) const
      {
      }
    };

    template <typename Model> class FlowBoundaries
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }

    public:
      template <int dim, typename NumberType, typename Solutions, typename M = Model>
      void boundary_numflux(std::array<Tensor<1, dim, NumberType>, M::Components::count_fe_functions(0)> &F,
                            const Tensor<1, dim> & /*normal*/, const Point<dim> &p, const Solutions &sol) const
      {
        static_assert(std::is_same<M, Model>::value, "Internal error: template parameter M must be the same as Model. "
                                                     "Do not explicitly specify the M template parameter.");
        asImp().flux(F, p, sol);
      }

      /// Batched boundary_numflux: the model's evaluate_batch at all points of a boundary batch, which
      /// requests only the flux.
      template <typename Out, typename Normals, typename Batch>
      void boundary_numflux_batch(Out &out, const Normals & /*normals*/, const Batch &batch) const
      {
        asImp().evaluate_batch(out, batch);
      }

      template <uint dependent, int dim, typename NumberType, typename Solutions, typename M = Model>
      void
      ldg_boundary_numflux(std::array<Tensor<1, dim, NumberType>, M::Components::count_fe_functions(dependent)> &BNF,
                           const Tensor<1, dim> & /*normal*/, const Point<dim> &p, const Solutions &u) const
      {
        static_assert(std::is_same<M, Model>::value, "Internal error: template parameter M must be the same as Model. "
                                                     "Do not explicitly specify the M template parameter.");
        asImp().template ldg_flux<dependent>(BNF, p, u);
      }

      /// Batched ldg_boundary_numflux: the model's ldg_evaluate_batch, which is asked for the flux only.
      template <uint dependent, typename Out, typename Normals, typename Batch>
      void ldg_boundary_numflux_batch(Out &out, const Normals & /*normals*/, const Batch &batch) const
      {
        asImp().template ldg_evaluate_batch<dependent>(out, batch);
      }
    };
  } // namespace def
} // namespace DiFfRG
