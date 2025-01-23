#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/loop_integrals.hh>

// external libraries
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/real.hpp>
#include <deal.II/base/quadrature_lib.h>

namespace DiFfRG
{
  using namespace dealii;

  class FlowEquations
  {
  protected:
    FlowEquations(const JSONValue &json, std::function<double(double)> optimize_x);

  public:
    void set_k(const double k);

    /**
     * @brief Print all deduced extents to the given spdlog.
     *
     */
    void print_parameters(const std::string logname) const;

    template <typename NT, int d, typename FUN> NT integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::integrate<NT, d>(fun, jac_x_quadrature, x_extent, k);
      else
        return LoopIntegrals::integrate<NT, d>(fun, x_quadrature, x_extent, k);
    }
    template <typename NT, int d, typename FUN> NT angle_integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT two_angle_integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::two_angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::two_angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT three_angle_integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::three_angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::three_angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }

    template <typename NT, int d, typename FUN> NT integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::integrate<NT, d>(fun, jac_x_quadrature, x_extent, k);
      else
        return LoopIntegrals::integrate<NT, d>(fun, x_quadrature, x_extent, k);
    }
    template <typename NT, int d, typename FUN> NT angle_integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT two_angle_integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::two_angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::two_angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT three_angle_integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::three_angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::three_angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }

    void set_jacobian_quadrature_factor(const double jacobian_quadrature_factor);

  protected:
    void update_x_extent();

    const uint x_quadrature_order;
    const QGauss<1> x_quadrature;
    double x_extent;

    const uint angle_quadrature_order;
    const QGauss<1> angle_quadrature;

    double jacobian_quadrature_factor;

    uint jac_x_quadrature_order;
    QGauss<1> jac_x_quadrature;
    uint jac_angle_quadrature_order;
    QGauss<1> jac_angle_quadrature;

    const double x_extent_tolerance;

    const std::function<double(double)> optimize_x;

    double k;
    bool unoptimized;
    const int verbosity;
  };

  class FlowEquationsFiniteT
  {
  protected:
    FlowEquationsFiniteT(const JSONValue &json, const double T, std::function<double(double)> optimize_x,
                         std::function<double(double)> optimize_x0, std::function<double(double)> optimize_q0);

  public:
    void set_k(const double k);

    /**
     * @brief Print all deduced extents to the given spdlog.
     *
     */
    void print_parameters(const std::string logname) const;

    template <typename NT, int d, typename FUN> NT integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::integrate<NT, d>(fun, jac_x_quadrature, x_extent, k);
      else
        return LoopIntegrals::integrate<NT, d>(fun, x_quadrature, x_extent, k);
    }
    template <typename NT, int d, typename FUN> NT angle_integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT two_angle_integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::two_angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::two_angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT three_angle_integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::three_angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::three_angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT spatial_integrate_and_integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::spatial_integrate_and_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k,
                                                                     jac_q0_quadrature, q0_extent);
      else
        return LoopIntegrals::spatial_integrate_and_integrate<NT, d>(fun, x_quadrature, x_extent, k, q0_quadrature,
                                                                     q0_extent);
    }
    template <typename NT, int d, typename FUN> NT spatial_angle_integrate_and_integrate(const FUN &fun) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::spatial_angle_integrate_and_integrate<NT, d>(
            fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature, jac_q0_quadrature, q0_extent);
      else
        return LoopIntegrals::spatial_angle_integrate_and_integrate<NT, d>(fun, x_quadrature, x_extent, k,
                                                                           angle_quadrature, q0_quadrature, q0_extent);
    }
    template <typename NT, int d, typename FUN> NT spatial_integrate_and_sum(const FUN &fun, const double T) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::spatial_integrate_and_sum<NT, d>(fun, jac_x_quadrature, x_extent, k, q0_summands,
                                                               jac_q0_quadrature, q0_extent, T);
      else
        return LoopIntegrals::spatial_integrate_and_sum<NT, d>(fun, x_quadrature, x_extent, k, q0_summands,
                                                               q0_quadrature, q0_extent, T);
    }
    template <typename NT, int d, typename FUN> NT spatial_angle_integrate_and_sum(const FUN &fun, const double T) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::spatial_angle_integrate_and_sum<NT, d>(
            fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature, q0_summands, jac_q0_quadrature, q0_extent, T);
      else
        return LoopIntegrals::spatial_angle_integrate_and_sum<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature,
                                                                     q0_summands, q0_quadrature, q0_extent, T);
    }
    template <typename NT, typename FUN> NT sum(const FUN &fun, const double T) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::sum<NT>(fun, q0_summands, jac_q0_quadrature, q0_extent, T);
      else
        return LoopIntegrals::sum<NT>(fun, q0_summands, q0_quadrature, q0_extent, T);
    }

    template <typename NT, int d, typename FUN> NT integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::integrate<NT, d>(fun, jac_x_quadrature, x_extent, k);
      else
        return LoopIntegrals::integrate<NT, d>(fun, x_quadrature, x_extent, k);
    }
    template <typename NT, int d, typename FUN> NT angle_integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT two_angle_integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::two_angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::two_angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT three_angle_integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::three_angle_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature);
      else
        return LoopIntegrals::three_angle_integrate<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature);
    }
    template <typename NT, int d, typename FUN> NT spatial_integrate_and_integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::spatial_integrate_and_integrate<NT, d>(fun, jac_x_quadrature, x_extent, k,
                                                                     jac_q0_quadrature, q0_extent);
      else
        return LoopIntegrals::spatial_integrate_and_integrate<NT, d>(fun, x_quadrature, x_extent, k, q0_quadrature,
                                                                     q0_extent);
    }
    template <typename NT, int d, typename FUN>
    NT spatial_angle_integrate_and_integrate(const FUN &fun, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::spatial_angle_integrate_and_integrate<NT, d>(
            fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature, jac_q0_quadrature, q0_extent);
      else
        return LoopIntegrals::spatial_angle_integrate_and_integrate<NT, d>(fun, x_quadrature, x_extent, k,
                                                                           angle_quadrature, q0_quadrature, q0_extent);
    }
    template <typename NT, int d, typename FUN>
    NT spatial_integrate_and_sum(const FUN &fun, const double T, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::spatial_integrate_and_sum<NT, d>(fun, jac_x_quadrature, x_extent, k, q0_summands,
                                                               jac_q0_quadrature, q0_extent, T);
      else
        return LoopIntegrals::spatial_integrate_and_sum<NT, d>(fun, x_quadrature, x_extent, k, q0_summands,
                                                               q0_quadrature, q0_extent, T);
    }
    template <typename NT, int d, typename FUN>
    NT spatial_angle_integrate_and_sum(const FUN &fun, const double T, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::spatial_angle_integrate_and_sum<NT, d>(
            fun, jac_x_quadrature, x_extent, k, jac_angle_quadrature, q0_summands, jac_q0_quadrature, q0_extent, T);
      else
        return LoopIntegrals::spatial_angle_integrate_and_sum<NT, d>(fun, x_quadrature, x_extent, k, angle_quadrature,
                                                                     q0_summands, q0_quadrature, q0_extent, T);
    }
    template <typename NT, typename FUN> NT sum(const FUN &fun, const double T, const double k) const
    {
      if constexpr (std::is_same_v<NT, autodiff::real> || std::is_same_v<NT, autodiff::dual>)
        return LoopIntegrals::sum<NT>(fun, q0_summands, jac_q0_quadrature, q0_extent, T);
      else
        return LoopIntegrals::sum<NT>(fun, q0_summands, q0_quadrature, q0_extent, T);
    }

    void set_jacobian_quadrature_factor(const double jacobian_quadrature_factor);

  protected:
    void update_x_extent();
    void update_x0_extent();
    void update_q0_extent();

    const uint x_quadrature_order;
    const QGauss<1> x_quadrature;
    double x_extent;

    const uint x0_summands;
    const uint x0_quadrature_order;
    const QGauss<1> x0_quadrature;
    double x0_extent;

    const uint q0_summands;
    const uint q0_quadrature_order;
    const QGauss<1> q0_quadrature;
    double q0_extent;

    const uint angle_quadrature_order;
    const QGauss<1> angle_quadrature;

    double jacobian_quadrature_factor;

    uint jac_x_quadrature_order;
    QGauss<1> jac_x_quadrature;
    uint jac_x0_quadrature_order;
    QGauss<1> jac_x0_quadrature;
    uint jac_q0_quadrature_order;
    QGauss<1> jac_q0_quadrature;
    uint jac_angle_quadrature_order;
    QGauss<1> jac_angle_quadrature;

    const double x_extent_tolerance;
    const double x0_extent_tolerance;
    const double q0_extent_tolerance;

    const double T;

    const std::function<double(double)> optimize_x;
    const std::function<double(double)> optimize_x0;
    const std::function<double(double)> optimize_q0;

    double k;
    bool unoptimized;
    const int verbosity;
  };
} // namespace DiFfRG