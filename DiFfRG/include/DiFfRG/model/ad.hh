#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/utils.hh>

// external libraries
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/real.hpp>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <tbb/tbb.h>

namespace DiFfRG
{
  namespace def
  {
    using namespace dealii;

    namespace internal
    {
      template <typename AD_type> struct AD_tools;

      template <> struct AD_tools<autodiff::dual> {
        template <uint n, typename Vector> static std::array<autodiff::dual, n> vector_to_AD(const Vector &v)
        {
          std::array<autodiff::dual, n> x;
          for (uint i = 0; i < n; ++i) {
            x[i] = v[i];
          }
          return x;
        }

        template <uint n, int dim, int r, typename NT>
        static std::array<dealii::Tensor<r, dim, autodiff::dual>, n>
        ten_to_AD(const std::vector<dealii::Tensor<r, dim, NT>> &v)
        {
          static_assert(r >= 1 && r <= 2, "Only rank 1 and 2 tensors are supported.");
          std::array<dealii::Tensor<r, dim, autodiff::dual>, n> x;
          for (uint i = 0; i < n; ++i) {
            if constexpr (r == 1)
              for (uint d = 0; d < dim; ++d)
                x[i][d] = v[i][d];
            else if constexpr (r == 2)
              for (uint d1 = 0; d1 < dim; ++d1)
                for (uint d2 = 0; d2 < dim; ++d2) {
                  x[i][d1][d2] = v[i][d1][d2];
                }
          }
          return x;
        }
      };

      template <> struct AD_tools<autodiff::real> {
        template <uint n, typename Vector> static std::array<autodiff::real, n> vector_to_AD(const Vector &v)
        {
          std::array<autodiff::real, n> x;
          for (uint i = 0; i < n; ++i) {
            x[i] = v[i];
          }
          return x;
        }

        template <uint n, int dim, int r, typename NT>
        static std::array<dealii::Tensor<r, dim, autodiff::real>, n>
        ten_to_AD(const std::vector<dealii::Tensor<r, dim, NT>> &v)
        {
          static_assert(r >= 1 && r <= 2, "Only rank 1 and 2 tensors are supported.");
          std::array<dealii::Tensor<r, dim, autodiff::real>, n> x;
          for (uint i = 0; i < n; ++i) {
            if constexpr (r == 1)
              for (uint d = 0; d < dim; ++d)
                x[i][d] = v[i][d];
            else if constexpr (r == 2)
              for (uint d1 = 0; d1 < dim; ++d1)
                for (uint d2 = 0; d2 < dim; ++d2) {
                  x[i][d1][d2] = v[i][d1][d2];
                }
          }
          return x;
        }
      };
    } // namespace internal

    template <typename Model, typename AD_type = autodiff::real> class ADjacobian_flux
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_grad(SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NT>>, n_to, n_from> &jF, const Point<dim> &p,
                              const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_fe_functions());

        const auto &u = std::get<tup_idx>(sol);
        const auto _du = AD_tools::template ten_to_AD<n_from>(u);
        tbb::parallel_for(
            tbb::blocked_range2d<uint>(0, Components::count_fe_functions(), 0, dim),
            [&](const tbb::blocked_range2d<uint> &r) {
              auto du = _du;
              for (uint j = r.rows().begin(); j < r.rows().end(); ++j) {
                for (uint d = r.cols().begin(); d < r.cols().end(); ++d) {
                  std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
                  // take derivative with respect to jth variable
                  seed(du[j][d]);
                  asImp().flux(res, p,
                               Vector::as(std::tuple_cat(tuple_first<tup_idx>(sol), std::tie(du),
                                                         tuple_last<std::tuple_size_v<Vector> - tup_idx - 1>(sol))));
                  for (uint i = 0; i < Components::count_fe_functions(); ++i) {
                    for (uint dd = 0; dd < dim; ++dd) {
                      jF(i, j)[dd][d] = grad(res[i][dd]);
                    }
                  }
                  unseed(du[j][d]);
                }
              }
            });
      }

      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_hess(SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NT>>, n_to, n_from> &jF, const Point<dim> &p,
                              const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_fe_functions());

        const auto &u = std::get<tup_idx>(sol);
        const auto _du = AD_tools::template ten_to_AD<n_from>(u);
        tbb::parallel_for(
            tbb::blocked_range3d<uint>(0, Components::count_fe_functions(), 0, dim, 0, dim),
            [&](const tbb::blocked_range3d<uint> &r) {
              auto du = _du;
              for (uint j = r.pages().begin(); j < r.pages().end(); ++j) {
                for (uint d1 = r.rows().begin(); d1 < r.rows().end(); ++d1)
                  for (uint d2 = r.cols().begin(); d2 < r.cols().end(); ++d2) {
                    std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
                    // take derivative with respect to jth variable
                    seed(du[j][d1][d2]);
                    asImp().flux(res, p,
                                 Vector::as(std::tuple_cat(tuple_first<tup_idx>(sol), std::tie(du),
                                                           tuple_last<std::tuple_size_v<Vector> - tup_idx - 1>(sol))));
                    for (uint i = 0; i < Components::count_fe_functions(); ++i) {
                      for (uint d = 0; d < dim; ++d) {
                        jF(i, j)[d][d1][d2] = grad(res[i][d]);
                      }
                    }
                    unseed(du[j][d1][d2]);
                  }
              }
            });
      }

      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_extr(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &jF, const Point<dim> &p,
                              const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_extractors());

        const auto &e = std::get<tup_idx>(sol);
        auto de = AD_tools::template vector_to_AD<n_from>(e);
        for (uint j = 0; j < Components::count_extractors(); ++j) {
          std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
          // take derivative with respect to jth variable
          seed(de[j]);
          asImp().flux(res, p,
                       Vector::as(std::tuple_cat(tuple_first<tup_idx>(sol), std::tie(de),
                                                 tuple_last<std::tuple_size_v<Vector> - tup_idx - 1>(sol))));
          for (uint i = 0; i < Components::count_fe_functions(); ++i) {
            for (uint d = 0; d < dim; ++d) {
              jF(i, j)[d] = grad(res[i][d]);
            }
          }
          unseed(de[j]);
        }
      }

      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &jF, const Point<dim> &p,
                         const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions(to));
        static_assert(n_from == Components::count_fe_functions(from));

        if constexpr (to == 0) {
          const auto &u = std::get<from>(sol);
          const auto _du = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(u);
          tbb::parallel_for(tbb::blocked_range<uint>(0, Components::count_fe_functions(from)),
                            [&](const tbb::blocked_range<uint> &r) {
                              auto du = _du;
                              for (uint j = r.begin(); j < r.end(); ++j) {
                                std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions(to)> res{{}};
                                // take derivative with respect to jth variable
                                seed(du[j]);
                                asImp().flux(
                                    res, p,
                                    Vector::as(std::tuple_cat(tuple_first<from>(sol), std::tie(du),
                                                              tuple_last<std::tuple_size_v<Vector> - from - 1>(sol))));
                                for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
                                  for (uint d = 0; d < dim; ++d) {
                                    jF(i, j)[d] = grad(res[i][d]);
                                  }
                                }
                                unseed(du[j]);
                              }
                            });
        } else {
          auto du = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(sol);
          for (uint j = 0; j < Components::count_fe_functions(from); ++j) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions(to)> res{{}};
            // take derivative with respect to jth variable
            seed(du[j]);
            asImp().template ldg_flux<to>(res, p, du);
            for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
              for (uint d = 0; d < dim; ++d) {
                jF(i, j)[d] = grad(res[i][d]);
              }
            }
            unseed(du[j]);
          }
        }
      }
    };

    template <typename Model, typename AD_type = autodiff::real> class ADjacobian_source
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source_grad(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &jS, const Point<dim> &p,
                                const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_fe_functions());

        const auto &u = std::get<tup_idx>(sol);
        const auto _du = AD_tools::template ten_to_AD<n_from>(u);
        tbb::parallel_for(
            tbb::blocked_range2d<uint>(0, Components::count_fe_functions(), 0, dim),
            [&](const tbb::blocked_range2d<uint> &r) {
              auto du = _du;
              for (uint j = r.rows().begin(); j < r.rows().end(); ++j) {
                for (uint d = r.cols().begin(); d < r.cols().end(); ++d) {
                  std::array<AD_type, Components::count_fe_functions()> res{{}};
                  // take derivative with respect to jth variable
                  seed(du[j][d]);
                  asImp().source(res, p,
                                 Vector::as(std::tuple_cat(tuple_first<tup_idx>(sol), std::tie(du),
                                                           tuple_last<std::tuple_size_v<Vector> - tup_idx - 1>(sol))));
                  for (uint i = 0; i < Components::count_fe_functions(); ++i) {
                    jS(i, j)[d] = grad(res[i]);
                  }
                  unseed(du[j][d]);
                }
              }
            });
      }

      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source_hess(SimpleMatrix<Tensor<2, dim, NT>, n_to, n_from> &jS, const Point<dim> &p,
                                const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_fe_functions());

        const auto &u = std::get<tup_idx>(sol);
        const auto _du = AD_tools::template ten_to_AD<n_from>(u);
        tbb::parallel_for(tbb::blocked_range3d<uint>(0, Components::count_fe_functions(), 0, dim, 0, dim),
                          [&](const tbb::blocked_range3d<uint> &r) {
                            auto du = _du;
                            for (uint j = r.pages().begin(); j < r.pages().end(); ++j) {
                              for (uint d1 = r.rows().begin(); d1 < r.rows().end(); ++d1) {
                                for (uint d2 = r.cols().begin(); d2 < r.cols().end(); ++d2) {
                                  std::array<AD_type, Components::count_fe_functions()> res{{}};
                                  // take derivative with respect to jth variable
                                  seed(du[j][d1][d2]);
                                  asImp().source(res, p,
                                                 Vector::as(std::tuple_cat(
                                                     tuple_first<tup_idx>(sol), std::tie(du),
                                                     tuple_last<std::tuple_size_v<Vector> - tup_idx - 1>(sol))));
                                  for (uint i = 0; i < Components::count_fe_functions(); ++i) {
                                    jS(i, j)[d1][d2] = grad(res[i]);
                                  }
                                  unseed(du[j][d1][d2]);
                                }
                              }
                            }
                          });
      }

      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source_extr(SimpleMatrix<NT, n_to, n_from> &jS, const Point<dim> &p, const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_extractors());

        const auto &e = std::get<tup_idx>(sol);
        auto de = AD_tools::template vector_to_AD<n_from>(e);
        for (uint j = 0; j < n_from; ++j) {
          std::array<AD_type, Components::count_fe_functions()> res{{}};
          // take derivative with respect to jth variable
          seed(de[j]);
          asImp().source(res, p,
                         Vector::as(std::tuple_cat(tuple_first<tup_idx>(sol), std::tie(de),
                                                   tuple_last<std::tuple_size_v<Vector> - tup_idx - 1>(sol))));
          for (uint i = 0; i < Components::count_fe_functions(); ++i) {
            jS(i, j) = grad(res[i]);
          }
          unseed(de[j]);
        }
      }

      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source(SimpleMatrix<NT, n_to, n_from> &jS, const Point<dim> &p, const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions(to));
        static_assert(n_from == Components::count_fe_functions(from));

        if constexpr (to == 0) {
          const auto &u = std::get<from>(sol);
          const auto _du = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(u);
          tbb::parallel_for(tbb::blocked_range<uint>(0, Components::count_fe_functions(from)),
                            [&](const tbb::blocked_range<uint> &r) {
                              auto du = _du;
                              for (uint j = r.begin(); j < r.end(); ++j) {
                                std::array<AD_type, Components::count_fe_functions(to)> res{{}};
                                // take derivative with respect to jth variable
                                seed(du[j]);
                                asImp().source(
                                    res, p,
                                    Vector::as(std::tuple_cat(tuple_first<from>(sol), std::tie(du),
                                                              tuple_last<std::tuple_size_v<Vector> - from - 1>(sol))));
                                for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
                                  jS(i, j) = grad(res[i]);
                                }
                                unseed(du[j]);
                              }
                            });
        } else {
          auto du = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(sol);
          for (uint j = 0; j < Components::count_fe_functions(from); ++j) {
            std::array<AD_type, Components::count_fe_functions(to)> res{{}};
            // take derivative with respect to jth variable
            seed(du[j]);
            asImp().template ldg_source<to>(res, p, du);
            for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
              jS(i, j) = grad(res[i]);
            }
            unseed(du[j]);
          }
        }
      }
    };

    template <typename Model, typename AD_type = autodiff::real> class ADjacobian_numflux
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector_s, typename Vector_n>
      void jacobian_numflux_grad(std::array<SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NT>>, n_to, n_from>, 2> &jNF,
                                 const Tensor<1, dim> &normal, const Point<dim> &p, const Vector_s &sol_s,
                                 const Vector_n &sol_n) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_fe_functions());

        const auto &u_s = std::get<tup_idx>(sol_s);
        const auto &u_n = std::get<tup_idx>(sol_n);

        auto du_s = AD_tools::template ten_to_AD<n_from>(u_s);
        for (uint j = 0; j < Components::count_fe_functions(); ++j) {
          for (uint d = 0; d < dim; ++d) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
            // take derivative with respect to jth variable
            seed(du_s[j][d]);
            asImp().numflux(res, normal, p,
                            Vector_s::as(std::tuple_cat(tuple_first<tup_idx>(sol_s), std::tie(du_s),
                                                        tuple_last<std::tuple_size_v<Vector_s> - tup_idx - 1>(sol_s))),
                            sol_n);
            for (uint i = 0; i < Components::count_fe_functions(); ++i) {
              for (uint dd = 0; dd < dim; ++dd) {
                jNF[0](i, j)[dd][d] = grad(res[i][dd]);
              }
            }
            unseed(du_s[j][d]);
          }
        }
        auto du_n = AD_tools::template ten_to_AD<n_from>(u_n);
        for (uint j = 0; j < Components::count_fe_functions(); ++j) {
          for (uint d = 0; d < dim; ++d) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
            // take derivative with respect to jth variable
            seed(du_n[j][d]);
            asImp().numflux(res, normal, p, sol_s,
                            Vector_n::as(std::tuple_cat(tuple_first<tup_idx>(sol_n), std::tie(du_n),
                                                        tuple_last<std::tuple_size_v<Vector_n> - tup_idx - 1>(sol_n))));
            for (uint i = 0; i < Components::count_fe_functions(); ++i) {
              for (uint dd = 0; dd < dim; ++dd) {
                jNF[1](i, j)[dd][d] = grad(res[i][dd]);
              }
            }
            unseed(du_n[j][d]);
          }
        }
      }

      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector_s, typename Vector_n>
      void jacobian_numflux_hess(std::array<SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NT>>, n_to, n_from>, 2> &jNF,
                                 const Tensor<1, dim> &normal, const Point<dim> &p, const Vector_s &sol_s,
                                 const Vector_n &sol_n) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_fe_functions());

        const auto &u_s = std::get<tup_idx>(sol_s);
        const auto &u_n = std::get<tup_idx>(sol_n);

        auto du_s = AD_tools::template ten_to_AD<n_from>(u_s);
        for (uint j = 0; j < Components::count_fe_functions(); ++j) {
          for (uint d1 = 0; d1 < dim; ++d1)
            for (uint d2 = 0; d2 < dim; ++d2) {
              std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
              // take derivative with respect to jth variable
              seed(du_s[j][d1][d2]);
              asImp().numflux(
                  res, normal, p,
                  Vector_s::as(std::tuple_cat(tuple_first<tup_idx>(sol_s), std::tie(du_s),
                                              tuple_last<std::tuple_size_v<Vector_s> - tup_idx - 1>(sol_s))),
                  sol_n);
              for (uint i = 0; i < Components::count_fe_functions(); ++i) {
                for (uint d = 0; d < dim; ++d) {
                  jNF[0](i, j)[d][d1][d2] = grad(res[i][d]);
                }
              }
              unseed(du_s[j][d1][d2]);
            }
        }
        auto du_n = AD_tools::template ten_to_AD<n_from>(u_n);
        for (uint j = 0; j < Components::count_fe_functions(); ++j) {
          for (uint d1 = 0; d1 < dim; ++d1)
            for (uint d2 = 0; d2 < dim; ++d2) {
              std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
              // take derivative with respect to jth variable
              seed(du_n[j][d1][d2]);
              asImp().numflux(
                  res, normal, p, sol_s,
                  Vector_n::as(std::tuple_cat(tuple_first<tup_idx>(sol_n), std::tie(du_n),
                                              tuple_last<std::tuple_size_v<Vector_n> - tup_idx - 1>(sol_n))));
              for (uint i = 0; i < Components::count_fe_functions(); ++i) {
                for (uint d = 0; d < dim; ++d) {
                  jNF[1](i, j)[d][d1][d2] = grad(res[i][d]);
                }
              }
              unseed(du_n[j][d1][d2]);
            }
        }
      }

      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector_s, typename Vector_n>
      void jacobian_numflux_extr(std::array<SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from>, 2> &jNF,
                                 const Tensor<1, dim> &normal, const Point<dim> &p, const Vector_s &sol_s,
                                 const Vector_n &sol_n) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_extractors());

        const auto &e = std::get<tup_idx>(sol_s);
        auto de = AD_tools::template vector_to_AD<n_from>(e);
        for (uint j = 0; j < n_from; ++j) {
          std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res_s{{}};
          std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res_n{{}};
          // take derivative with respect to jth variable
          seed(de[j]);
          asImp().numflux(res_s, normal, p,
                          Vector_s::as(std::tuple_cat(tuple_first<tup_idx>(sol_s), std::tie(de),
                                                      tuple_last<std::tuple_size_v<Vector_s> - tup_idx - 1>(sol_s))),
                          sol_n);
          asImp().numflux(res_n, normal, p, sol_s,
                          Vector_n::as(std::tuple_cat(tuple_first<tup_idx>(sol_n), std::tie(de),
                                                      tuple_last<std::tuple_size_v<Vector_n> - tup_idx - 1>(sol_n))));
          for (uint i = 0; i < Components::count_fe_functions(); ++i) {
            for (uint d = 0; d < dim; ++d) {
              jNF[0](i, j)[d] = grad(res_s[i][d]);
              jNF[1](i, j)[d] = grad(res_n[i][d]);
            }
          }
          unseed(de[j]);
        }
      }

      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector_s, typename Vector_n>
      void jacobian_numflux(std::array<SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from>, 2> &jNF,
                            const Tensor<1, dim> &normal, const Point<dim> &p, const Vector_s &sol_s,
                            const Vector_n &sol_n) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions(to));
        static_assert(n_from == Components::count_fe_functions(from));

        if constexpr (to == 0) {
          const auto &u_s = std::get<from>(sol_s);
          const auto &u_n = std::get<from>(sol_n);

          auto du_s = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(u_s);
          for (uint j = 0; j < Components::count_fe_functions(from); ++j) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions(to)> res{{}};
            // take derivative with respect to jth variable
            seed(du_s[j]);
            asImp().numflux(res, normal, p,
                            Vector_s::as(std::tuple_cat(tuple_first<from>(sol_s), std::tie(du_s),
                                                        tuple_last<std::tuple_size_v<Vector_s> - from - 1>(sol_s))),
                            sol_n);
            for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
              for (uint d = 0; d < dim; ++d) {
                jNF[0](i, j)[d] = grad(res[i][d]);
              }
            }
            unseed(du_s[j]);
          }
          auto du_n = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(u_n);
          for (uint j = 0; j < Components::count_fe_functions(from); ++j) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions(to)> res{{}};
            // take derivative with respect to jth variable
            seed(du_n[j]);
            asImp().numflux(res, normal, p, sol_s,
                            Vector_n::as(std::tuple_cat(tuple_first<from>(sol_n), std::tie(du_n),
                                                        tuple_last<std::tuple_size_v<Vector_n> - from - 1>(sol_n))));
            for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
              for (uint d = 0; d < dim; ++d) {
                jNF[1](i, j)[d] = grad(res[i][d]);
              }
            }
            unseed(du_n[j]);
          }
        } else {
          auto du_s = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(sol_s);
          for (uint j = 0; j < Components::count_fe_functions(from); ++j) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions(to)> res{{}};
            // take derivative with respect to jth variable
            seed(du_s[j]);
            asImp().template ldg_numflux<to>(res, normal, p, du_s, sol_n);
            for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
              for (uint d = 0; d < dim; ++d) {
                jNF[0](i, j)[d] = grad(res[i][d]);
              }
            }
            unseed(du_s[j]);
          }
          auto du_n = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(sol_n);
          for (uint j = 0; j < Components::count_fe_functions(from); ++j) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions(to)> res{{}};
            // take derivative with respect to jth variable
            seed(du_n[j]);
            asImp().template ldg_numflux<to>(res, normal, p, sol_s, du_n);
            for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
              for (uint d = 0; d < dim; ++d) {
                jNF[1](i, j)[d] = grad(res[i][d]);
              }
            }
            unseed(du_n[j]);
          }
        }
      }
    };

    template <typename Model, typename AD_type = autodiff::real> class ADjacobian_boundary_numflux
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_boundary_numflux_grad(SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NT>>, n_to, n_from> &jBNF,
                                          const Tensor<1, dim> &normal, const Point<dim> &p, const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_fe_functions());

        const auto &u = std::get<tup_idx>(sol);
        auto du = AD_tools::template ten_to_AD<n_from>(u);
        for (uint j = 0; j < Components::count_fe_functions(); ++j) {
          for (uint d = 0; d < dim; ++d) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
            // take derivative with respect to jth variable
            seed(du[j][d]);
            asImp().boundary_numflux(
                res, normal, p,
                Vector::as(std::tuple_cat(tuple_first<tup_idx>(sol), std::tie(du),
                                          tuple_last<std::tuple_size_v<Vector> - tup_idx - 1>(sol))));
            for (uint i = 0; i < Components::count_fe_functions(); ++i) {
              for (uint dd = 0; dd < dim; ++dd) {
                jBNF(i, j)[dd][d] = grad(res[i][dd]);
              }
            }
            unseed(du[j][d]);
          }
        }
      }

      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_boundary_numflux_hess(SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NT>>, n_to, n_from> &jBNF,
                                          const Tensor<1, dim> &normal, const Point<dim> &p, const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_fe_functions());

        const auto &u = std::get<tup_idx>(sol);
        auto du = AD_tools::template ten_to_AD<n_from>(u);
        for (uint j = 0; j < Components::count_fe_functions(); ++j) {
          for (uint d1 = 0; d1 < dim; ++d1)
            for (uint d2 = 0; d2 < dim; ++d2) {
              std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
              // take derivative with respect to jth variable
              seed(du[j][d1][d2]);
              asImp().boundary_numflux(
                  res, normal, p,
                  Vector::as(std::tuple_cat(tuple_first<tup_idx>(sol), std::tie(du),
                                            tuple_last<std::tuple_size_v<Vector> - tup_idx - 1>(sol))));
              for (uint i = 0; i < Components::count_fe_functions(); ++i) {
                for (uint d = 0; d < dim; ++d) {
                  jBNF(i, j)[d][d1][d2] = grad(res[i][d]);
                }
              }
              unseed(du[j][d1][d2]);
            }
        }
      }

      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_boundary_numflux_extr(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &jBNF,
                                          const Tensor<1, dim> &normal, const Point<dim> &p, const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions());
        static_assert(n_from == Components::count_extractors());

        const auto &e = std::get<tup_idx>(sol);
        auto de = AD_tools::template vector_to_AD<n_from>(e);
        for (uint j = 0; j < n_from; ++j) {
          std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions()> res{{}};
          // take derivative with respect to jth variable
          seed(de[j]);
          asImp().boundary_numflux(
              res, normal, p,
              Vector::as(std::tuple_cat(tuple_first<tup_idx>(sol), std::tie(de),
                                        tuple_last<std::tuple_size_v<Vector> - tup_idx - 1>(sol))));
          for (uint i = 0; i < Components::count_fe_functions(); ++i) {
            for (uint d = 0; d < dim; ++d) {
              jBNF(i, j)[d] = grad(res[i][d]);
            }
          }
          unseed(de[j]);
        }
      }

      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_boundary_numflux(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &jBNF, const Tensor<1, dim> &normal,
                                     const Point<dim> &p, const Vector &sol) const
      {
        using Components = typename Model::Components;
        static_assert(n_to == Components::count_fe_functions(to));
        static_assert(n_from == Components::count_fe_functions(from));

        if constexpr (to == 0) {
          const auto &u = std::get<from>(sol);
          auto du = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(u);
          for (uint j = 0; j < Components::count_fe_functions(from); ++j) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions(to)> res{{}};
            // take derivative with respect to jth variable
            seed(du[j]);
            asImp().boundary_numflux(res, normal, p,
                                     Vector::as(std::tuple_cat(tuple_first<from>(sol), std::tie(du),
                                                               tuple_last<std::tuple_size_v<Vector> - from - 1>(sol))));
            for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
              for (uint d = 0; d < dim; ++d) {
                jBNF(i, j)[d] = grad(res[i][d]);
              }
            }
            unseed(du[j]);
          }
        } else {
          auto du = AD_tools::template vector_to_AD<Components::count_fe_functions(from)>(sol);
          for (uint j = 0; j < Components::count_fe_functions(from); ++j) {
            std::array<Tensor<1, dim, AD_type>, Components::count_fe_functions(to)> res{{}};
            // take derivative with respect to jth variable
            seed(du[j]);
            asImp().template ldg_boundary_numflux<to>(res, normal, p, du);
            for (uint i = 0; i < Components::count_fe_functions(to); ++i) {
              for (uint d = 0; d < dim; ++d) {
                jBNF(i, j)[d] = grad(res[i][d]);
              }
            }
            unseed(du[j]);
          }
        }
      }
    };

    template <typename Model, typename AD_type = autodiff::real> class ADjacobian_mass
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint dot, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_mass(SimpleMatrix<NT, n_to, n_from> &jM, const Point<dim> &p, const Vector &u,
                         const Vector &u_dot) const
      {
        using Components = typename Model::Components;
        static_assert(n_from == Components::count_fe_functions() && n_to == Components::count_fe_functions());

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

    template <typename Model, typename AD_type = autodiff::real> class ADjacobian_variables
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint to, typename NT, typename Solution>
      void jacobian_variables(FullMatrix<NT> &jac, const Solution &sol) const
      {
        const auto &variables = std::get<0>(sol);
        const auto &extractors = std::get<1>(sol);
        if constexpr (to == 0) {
          AssertThrow(jac.m() == variables.size() && jac.n() == variables.size(),
                      ExcMessage("Assure that the jacobian has the right dimension!"));
        } else if constexpr (to == 1) {
          AssertThrow(jac.m() == variables.size() && jac.n() == extractors.size(),
                      ExcMessage("Assure that the jacobian has the right dimension!"));
        }

        if constexpr (to == 0) {
          auto du = AD_tools::template vector_to_AD<Model::Components::count_variables()>(variables);
          for (uint j = 0; j < Model::Components::count_variables(); ++j) {
            std::array<AD_type, Model::Components::count_variables()> res{{}};
            // take derivative with respect to jth variable
            seed(du[j]);
            asImp().dt_variables(res,
                                 Solution::as(std::tuple_cat(tuple_first<to>(sol), std::tie(du),
                                                             tuple_last<std::tuple_size_v<Solution> - to - 1>(sol))));
            for (uint i = 0; i < Model::Components::count_variables(); ++i) {
              jac(i, j) = grad(res[i]);
            }
            unseed(du[j]);
          }
        } else if constexpr (to == 1) {
          auto du = AD_tools::template vector_to_AD<Model::Components::count_extractors()>(extractors);
          for (uint j = 0; j < Model::Components::count_extractors(); ++j) {
            std::array<AD_type, Model::Components::count_variables()> res{{}};
            // take derivative with respect to jth variable
            seed(du[j]);
            asImp().dt_variables(res,
                                 Solution::as(std::tuple_cat(tuple_first<to>(sol), std::tie(du),
                                                             tuple_last<std::tuple_size_v<Solution> - to - 1>(sol))));
            for (uint i = 0; i < Model::Components::count_extractors(); ++i) {
              jac(i, j) = grad(res[i]);
            }
            unseed(du[j]);
          }
        }
      }
    };

    template <typename Model, typename AD_type = autodiff::real> class ADjacobian_extractors
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }
      using AD_tools = internal::AD_tools<AD_type>;

    public:
      template <uint to, int dim, typename NT, typename Solution>
      void jacobian_extractors(FullMatrix<NT> &jac, const Point<dim> &x, const Solution &sol) const
      {
        static_assert(std::is_same_v<NT, double>, "Only double is supported for now!");
        const auto &fe_functions = std::get<0>(sol);
        const auto &fe_derivatives = std::get<1>(sol);
        const auto &fe_hessians = std::get<2>(sol);

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
          for (uint j = 0; j < Model::Components::count_fe_functions(); ++j) {
            std::array<AD_type, Model::Components::count_extractors()> res{{}};
            // take derivative with respect to jth variable
            seed(du[j]);
            asImp().extract(res, x,
                            Solution::as(std::tuple_cat(tuple_first<to>(sol), std::tie(du),
                                                        tuple_last<std::tuple_size_v<Solution> - to - 1>(sol))));
            for (uint i = 0; i < Model::Components::count_extractors(); ++i) {
              jac(i, j) = grad(res[i]);
            }
            unseed(du[j]);
          }
        } else if constexpr (to == 1) {
          auto du = AD_tools::template ten_to_AD<Model::Components::count_fe_functions()>(fe_derivatives);
          for (uint j = 0; j < Model::Components::count_fe_functions(); ++j) {
            for (uint d1 = 0; d1 < dim; ++d1) {
              std::array<AD_type, Model::Components::count_extractors()> res{{}};
              // take derivative with respect to jth variable
              seed(du[j][d1]);
              asImp().extract(res, x,
                              Solution::as(std::tuple_cat(tuple_first<to>(sol), std::tie(du),
                                                          tuple_last<std::tuple_size_v<Solution> - to - 1>(sol))));
              for (uint i = 0; i < Model::Components::count_extractors(); ++i) {
                jac(i, j * dim + d1) = grad(res[i]);
              }
              unseed(du[j][d1]);
            }
          }
        } else if constexpr (to == 2) {
          auto du = AD_tools::template ten_to_AD<Model::Components::count_fe_functions()>(fe_hessians);
          for (uint j = 0; j < Model::Components::count_fe_functions(); ++j) {
            for (uint d1 = 0; d1 < dim; ++d1)
              for (uint d2 = 0; d2 < dim; ++d2) {
                std::array<AD_type, Model::Components::count_extractors()> res{{}};
                // take derivative with respect to jth variable
                seed(du[j][d1][d2]);
                asImp().extract(res, x,
                                Solution::as(std::tuple_cat(tuple_first<to>(sol), std::tie(du),
                                                            tuple_last<std::tuple_size_v<Solution> - to - 1>(sol))));
                for (uint i = 0; i < Model::Components::count_extractors(); ++i) {
                  jac(i, j * dim * dim + d1 * dim + d2) = grad(res[i]);
                }
                unseed(du[j][d1][d2]);
              }
          }
        }
      }
    };

    template <typename Model>
    class AD_real : public ADjacobian_flux<Model, autodiff::real>,
                    public ADjacobian_source<Model, autodiff::real>,
                    public ADjacobian_numflux<Model, autodiff::real>,
                    public ADjacobian_boundary_numflux<Model, autodiff::real>,
                    public ADjacobian_mass<Model, autodiff::real>,
                    public ADjacobian_variables<Model, autodiff::real>,
                    public ADjacobian_extractors<Model, autodiff::real>
    {
    };

    template <typename Model>
    class AD_dual : public ADjacobian_flux<Model, autodiff::dual>,
                    public ADjacobian_source<Model, autodiff::dual>,
                    public ADjacobian_numflux<Model, autodiff::dual>,
                    public ADjacobian_boundary_numflux<Model, autodiff::dual>,
                    public ADjacobian_mass<Model, autodiff::dual>,
                    public ADjacobian_variables<Model, autodiff::dual>,
                    public ADjacobian_extractors<Model, autodiff::dual>
    {
    };

    template <typename Model> using AD = AD_real<Model>;

    template <typename Model>
    class FE_AD : public ADjacobian_flux<Model, autodiff::real>,
                  public ADjacobian_source<Model, autodiff::real>,
                  public ADjacobian_numflux<Model, autodiff::real>,
                  public ADjacobian_boundary_numflux<Model, autodiff::real>,
                  public ADjacobian_mass<Model, autodiff::real>
    {
    public:
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_extr(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, const Point<dim> &,
                              const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source_extr(SimpleMatrix<NT, n_to, n_from> &, const Point<dim> &, const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector_s, typename Vector_n>
      void jacobian_numflux_extr(std::array<SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from>, 2> &,
                                 const Tensor<1, dim> &, const Point<dim> &, const Vector_s &, const Vector_n &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_boundary_numflux_extr(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, const Tensor<1, dim> &,
                                          const Point<dim> &, const Vector &) const
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

    class NoJacobians
    {
    public:
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_grad(SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NT>>, n_to, n_from> &, const Point<dim> &,
                              const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_hess(SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NT>>, n_to, n_from> &, const Point<dim> &,
                              const Vector &) const
      {
      }
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux_extr(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, const Point<dim> &,
                              const Vector &) const
      {
      }
      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_flux(SimpleMatrix<Tensor<1, dim, NT>, n_to, n_from> &, const Point<dim> &, const Vector &) const
      {
      }
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
      template <uint tup_idx, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source_extr(SimpleMatrix<NT, n_to, n_from> &, const Point<dim> &, const Vector &) const
      {
      }
      template <uint from, uint to, uint n_from, uint n_to, int dim, typename NT, typename Vector>
      void jacobian_source(SimpleMatrix<NT, n_to, n_from> &, const Point<dim> &, const Vector &) const
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