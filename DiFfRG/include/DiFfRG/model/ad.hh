#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/utils.hh>

// external libraries
#include <autodiff/forward/real.hpp>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/full_matrix.h>

namespace DiFfRG
{
  namespace def
  {
    using namespace dealii;
    using std::get;

    namespace internal
    {
      template <typename AD_type> struct AD_tools;

      template <> struct AD_tools<autodiff::real> {
        template <uint n, typename Vector> static std::array<autodiff::real, n> vector_to_AD(const Vector &v)
        {
          std::array<autodiff::real, n> x;
          for (uint i = 0; i < n; ++i)
            x[i] = v[i];
          return x;
        }

        // Any indexable container of dealii::Tensor works. Rank and dimension are read off the element type.
        template <uint n, typename Container> static auto ten_to_AD(const Container &v)
        {
          using TensorType = std::decay_t<decltype(v[0])>;
          constexpr int r = static_cast<int>(TensorType::rank);
          constexpr int dim = static_cast<int>(TensorType::dimension);
          static_assert(r >= 1 && r <= 2, "Only rank 1 and 2 tensors are supported.");
          std::array<dealii::Tensor<r, dim, autodiff::real>, n> x;
          for (uint i = 0; i < n; ++i) {
            if constexpr (r == 1) {
              for (uint d = 0; d < dim; ++d)
                x[i][d] = v[i][d];
            } else if constexpr (r == 2) {
              for (uint d1 = 0; d1 < dim; ++d1)
                for (uint d2 = 0; d2 < dim; ++d2) {
                  x[i][d1][d2] = v[i][d1][d2];
                }
            }
          }
          return x;
        }
      };
    } // namespace internal

    /**
     * @brief Marks a model whose flux, source and numerical-flux jacobians are the forward-mode AD of its
     * evaluate_batch (and of the numflux_batch / boundary_numflux_batch built on it), seed-stacked over all points
     * of a batch; see DiFfRG::internal::seed_stacked_jacobian. def::AD and def::FE_AD derive from it. A model
     * without it provides the per-point jacobian_* callbacks itself (or def::NoJacobians).
     */
    struct BatchADJacobians {
    };

    /**
     * @brief Marks a model whose extractors are frozen within a Newton step, i.e. whose jacobian_extractors is
     * zero (def::FE_AD): the AD jacobians then skip the extractor directions.
     */
    struct FrozenExtractors {
    };

    template <typename Model> class ADjacobian_mass
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_type = autodiff::real;
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint dot, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_mass(SimpleMatrix<NT, n_to, n_from> &jM, const Point<dim> &p, const Vector &u,
                         const Vector &u_dot) const
      {
        using Components = typename Model::Components;
        static_assert(n_from == Components::count_fe_functions() && n_to == Components::count_fe_functions(),
                      "jacobian_mass: n_from and n_to must both equal count_fe_functions()");

        if constexpr (dot == 0) {
          auto du = AD_tools::template vector_to_AD<Components::count_fe_functions(0)>(u);
          for (uint j = 0; j < Components::count_fe_functions(0); ++j) {
            std::array<AD_type, Components::count_fe_functions(0)> res{{}};
            // take derivative with respect to jth variable
            seed(du[j]);
            asImp().mass(res, p, du, u_dot);
            for (uint i = 0; i < Components::count_fe_functions(0); ++i) {
              jM(i, j) = grad(res[i]);
            }
            unseed(du[j]);
          }
        } else {
          auto du_dot = AD_tools::template vector_to_AD<Components::count_fe_functions(0)>(u_dot);
          for (uint j = 0; j < Components::count_fe_functions(0); ++j) {
            std::array<AD_type, Components::count_fe_functions(0)> res{{}};
            // take derivative with respect to jth variable
            seed(du_dot[j]);
            asImp().mass(res, p, u, du_dot);
            for (uint i = 0; i < Components::count_fe_functions(0); ++i) {
              jM(i, j) = grad(res[i]);
            }
            unseed(du_dot[j]);
          }
        }
      }
    };

    template <typename Model> class ADjacobian_variables
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_type = autodiff::real;
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint to, typename NT, typename Solution>
      void jacobian_variables(FullMatrix<NT> &jac, const Solution &sol) const
      {
        const auto &variables = get<0>(sol);
        const auto &extractors = get<1>(sol);
        if constexpr (to == 0) {
          AssertThrow(jac.m() == variables.size() && jac.n() == variables.size(),
                      ExcMessage("Assure that the jacobian has the right dimension!"));
        } else if constexpr (to == 1) {
          AssertThrow(jac.m() == variables.size() && jac.n() == extractors.size(),
                      ExcMessage("Assure that the jacobian has the right dimension!"));
        }

        if constexpr (to == 0) {
          auto du = AD_tools::template vector_to_AD<Model::Components::count_variables()>(variables);
          auto ad_sol = std::tuple_cat(tuple_first<to>(sol), std::tie(du), tuple_last<Solution::size - to - 1>(sol));
          for (uint j = 0; j < Model::Components::count_variables(); ++j) {
            std::array<AD_type, Model::Components::count_variables()> res{{}};
            seed(du[j]);
            asImp().dt_variables(res, Solution::as(ad_sol));
            for (uint i = 0; i < Model::Components::count_variables(); ++i) {
              jac(i, j) = grad(res[i]);
            }
            unseed(du[j]);
          }
        } else if constexpr (to == 1) {
          auto du = AD_tools::template vector_to_AD<Model::Components::count_extractors()>(extractors);
          auto ad_sol = std::tuple_cat(tuple_first<to>(sol), std::tie(du), tuple_last<Solution::size - to - 1>(sol));
          for (uint j = 0; j < Model::Components::count_extractors(); ++j) {
            std::array<AD_type, Model::Components::count_variables()> res{{}};
            seed(du[j]);
            asImp().dt_variables(res, Solution::as(ad_sol));
            for (uint i = 0; i < Model::Components::count_extractors(); ++i) {
              jac(i, j) = grad(res[i]);
            }
            unseed(du[j]);
          }
        }
      }
    };

    template <typename Model> class ADjacobian_extractors
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_type = autodiff::real;
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint to, int dim, typename NT, typename Solution>
      void jacobian_extractors(FullMatrix<NT> &jac, const Point<dim> &x, const Solution &sol) const
      {
        static_assert(std::is_same_v<NT, double>, "Only double is supported for now!");
        const auto &fe_functions = get<0>(sol);
        const auto &fe_derivatives = get<1>(sol);
        const auto &fe_hessians = get<2>(sol);

        if constexpr (to == 0) {
          AssertThrow(jac.m() == Model::Components::count_extractors() && jac.n() == fe_functions.size(),
                      ExcMessage("Assure that the jacobian has the right dimension!"));
        } else if constexpr (to == 1) {
          AssertThrow(jac.m() == Model::Components::count_extractors() && jac.n() == fe_derivatives.size() * dim,
                      ExcMessage("Assure that the jacobian has the right dimension!"));
        } else if constexpr (to == 2) {
          AssertThrow(jac.m() == Model::Components::count_extractors() && jac.n() == fe_derivatives.size() * dim * dim,
                      ExcMessage("Assure that the jacobian has the right dimension!"));
        }

        if constexpr (to == 0) {
          auto du = AD_tools::template vector_to_AD<Model::Components::count_fe_functions()>(fe_functions);
          auto ad_sol = std::tuple_cat(tuple_first<to>(sol), std::tie(du), tuple_last<Solution::size - to - 1>(sol));
          for (uint j = 0; j < Model::Components::count_fe_functions(); ++j) {
            std::array<AD_type, Model::Components::count_extractors()> res{{}};
            seed(du[j]);
            asImp().extract(res, x, Solution::as(ad_sol));
            for (uint i = 0; i < Model::Components::count_extractors(); ++i) {
              jac(i, j) = grad(res[i]);
            }
            unseed(du[j]);
          }
        } else if constexpr (to == 1) {
          auto du = AD_tools::template ten_to_AD<Model::Components::count_fe_functions()>(fe_derivatives);
          auto ad_sol = std::tuple_cat(tuple_first<to>(sol), std::tie(du), tuple_last<Solution::size - to - 1>(sol));
          for (uint j = 0; j < Model::Components::count_fe_functions(); ++j) {
            for (uint d1 = 0; d1 < dim; ++d1) {
              std::array<AD_type, Model::Components::count_extractors()> res{{}};
              seed(du[j][d1]);
              asImp().extract(res, x, Solution::as(ad_sol));
              for (uint i = 0; i < Model::Components::count_extractors(); ++i) {
                jac(i, j * dim + d1) = grad(res[i]);
              }
              unseed(du[j][d1]);
            }
          }
        } else if constexpr (to == 2) {
          auto du = AD_tools::template ten_to_AD<Model::Components::count_fe_functions()>(fe_hessians);
          auto ad_sol = std::tuple_cat(tuple_first<to>(sol), std::tie(du), tuple_last<Solution::size - to - 1>(sol));
          for (uint j = 0; j < Model::Components::count_fe_functions(); ++j) {
            for (uint d1 = 0; d1 < dim; ++d1)
              for (uint d2 = 0; d2 < dim; ++d2) {
                std::array<AD_type, Model::Components::count_extractors()> res{{}};
                seed(du[j][d1][d2]);
                asImp().extract(res, x, Solution::as(ad_sol));
                for (uint i = 0; i < Model::Components::count_extractors(); ++i) {
                  jac(i, j * dim * dim + d1 * dim + d2) = grad(res[i]);
                }
                unseed(du[j][d1][d2]);
              }
          }
        }
      }
    };

    /**
     * @brief All jacobians by forward-mode AD: flux, source and numerical fluxes seed-stacked over the batches
     * (BatchADJacobians), mass, variables and extractors per point.
     */
    template <typename Model>
    class AD : public BatchADJacobians,
               public ADjacobian_mass<Model>,
               public ADjacobian_variables<Model>,
               public ADjacobian_extractors<Model>
    {
    };

    /**
     * @brief As AD, but with frozen extractors and variables: their jacobians are zero, so the jacobian holds only
     * the direct dependence on the FE functions.
     */
    template <typename Model>
    class FE_AD : public BatchADJacobians, public FrozenExtractors, public ADjacobian_mass<Model>
    {
    public:
      template <uint to, int dim, typename NT, typename Solution>
      void jacobian_extractors(FullMatrix<NT> &, const Point<dim> &, const Solution &) const
      {
      }
      template <uint to, typename NT, typename Solution>
      void jacobian_variables(FullMatrix<NT> &, const Solution &) const
      {
      }
    };

    class NoJacobians
    {
    public:
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source_grad(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, const Point<dim> &,
                                const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source_hess(SimpleMatrix<Tensor<2, dim, NT>, n_to, n_from> &, const Point<dim> &,
                                const Vector &) const
      {
      }
      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source(SimpleMatrix<NT, n_to, n_from> &, const Point<dim> &, const Vector &) const
      {
      }
      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_source(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, SimpleMatrix<NT, n_to, n_from> &,
                                const Point<dim> &, const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_source_grad(SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NT>>, n_to, n_from> &,
                                     SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, const Point<dim> &,
                                     const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_source_hess(SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NT>>, n_to, n_from> &,
                                     SimpleMatrix<Tensor<2, dim, NT>, n_to, n_from> &, const Point<dim> &,
                                     const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_source_extr(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, SimpleMatrix<NT, n_to, n_from> &,
                                     const Point<dim> &, const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector_s, typename Vector_n>
      void jacobian_numflux_grad(std::array<SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NT>>, n_to, n_from>, 2> &,
                                 const Tensor<1, dim> &, const Point<dim> &, const Vector_s &, const Vector_n &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector_s, typename Vector_n>
      void jacobian_numflux_hess(std::array<SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NT>>, n_to, n_from>, 2> &,
                                 const Tensor<1, dim> &, const Point<dim> &, const Vector_s &, const Vector_n &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector_s, typename Vector_n>
      void jacobian_numflux_extr(std::array<SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from>, 2> &,
                                 const Tensor<1, dim> &, const Point<dim> &, const Vector_s &, const Vector_n &) const
      {
      }
      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector_s, typename Vector_n>
      void jacobian_numflux(std::array<SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from>, 2> &, const Tensor<1, dim> &,
                            const Point<dim> &, const Vector_s &, const Vector_n &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_boundary_numflux_grad(SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NT>>, n_to, n_from> &,
                                          const Tensor<1, dim> &, const Point<dim> &, const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_boundary_numflux_hess(SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NT>>, n_to, n_from> &,
                                          const Tensor<1, dim> &, const Point<dim> &, const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_boundary_numflux_extr(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, const Tensor<1, dim> &,
                                          const Point<dim> &, const Vector &) const
      {
      }
      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_boundary_numflux(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, const Tensor<1, dim> &,
                                     const Point<dim> &, const Vector &) const
      {
      }
      template <uint dot, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_mass(SimpleMatrix<NT, n_to, n_from> &, const Point<dim> &, const Vector &, const Vector &) const
      {
      }

      template <uint to, int dim, typename NT, typename Solution>
      void jacobian_extractors(FullMatrix<NT> &, const Point<dim> &, const Solution &) const
      {
      }
      template <uint to, typename NT, typename Solution>
      void jacobian_variables(FullMatrix<NT> &, const Solution &) const
      {
      }
    };

  } // namespace def
} // namespace DiFfRG