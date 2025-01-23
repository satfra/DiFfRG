#pragma once

// standard library
#include <array>

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>

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
        static_assert(std::is_same<M, Model>::value, "The call of this method should not have touched M.");
        using Components = typename M::Components;

        std::array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> F_s{};
        std::array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> F_n{};
        asImp().flux(F_s, p, sol_s);
        asImp().flux(F_n, p, sol_n);

        const auto &u_s = std::get<0>(sol_s);
        const auto &u_n = std::get<0>(sol_n);

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
        static_assert(std::is_same<M, Model>::value, "The call of this method should not have touched M.");
        static_assert(dependent >= 1, "This is LDG, not DG.");

        using Dirs = typename C<dependent - 1>::template value<0>;
        using UD = typename C<dependent - 1>::template value<1>;
        static_assert(Dirs::size == UD::size && UD::size >= M::Components::count_fe_functions(dependent),
                      "Mismatch in array sizes.");
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
    };

    template <typename Model> class NoNumFlux
    {
    public:
      template <int dim, typename NumberType, typename Solutions_s, typename Solutions_n, typename M = Model>
      void numflux(std::array<Tensor<1, dim, NumberType>, M::Components::count_fe_functions(0)> &,
                   const Tensor<1, dim> &, const Point<dim> &, const Solutions_s &, const Solutions_n &) const
      {
        static_assert(std::is_same<M, Model>::value, "The call of this method should not have touched M.");
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
        static_assert(std::is_same<M, Model>::value, "The call of this method should not have touched M.");
        asImp().flux(F, p, sol);
      }

      template <uint dependent, int dim, typename NumberType, typename Solutions, typename M = Model>
      void
      ldg_boundary_numflux(std::array<Tensor<1, dim, NumberType>, M::Components::count_fe_functions(dependent)> &BNF,
                           const Tensor<1, dim> & /*normal*/, const Point<dim> &p, const Solutions &u) const
      {
        static_assert(std::is_same<M, Model>::value, "The call of this method should not have touched M.");
        asImp().template ldg_flux<dependent>(BNF, p, u);
      }
    };
  } // namespace def
} // namespace DiFfRG
