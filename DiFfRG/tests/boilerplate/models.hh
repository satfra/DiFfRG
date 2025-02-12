#pragma once

// external libraries
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/model/model.hh>

namespace DiFfRG
{
  namespace Testing
  {
    using namespace dealii;

    struct PhysicalParameters {
      std::array<double, 3> initial_x0{{0., 0., 0.}};
      std::array<double, 3> initial_x1{{0., 0., 0.}};
      std::array<double, 3> initial_x2{{0., 0., 0.}};
      std::array<double, 3> initial_x3{{0., 0., 0.}};
    };

    template <uint components> struct compFactory {
      using value = ComponentDescriptor<FEFunctionDescriptor<>>;
    };

    template <> struct compFactory<1> {
      using value = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>;
    };

    template <> struct compFactory<2> {
      using value = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>>;
    };

    template <> struct compFactory<3> {
      using value = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">, Scalar<"w">>>;
    };

    template <uint dim, uint components = 1>
    class ModelConstant
        : public def::AbstractModel<ModelConstant<dim, components>, typename compFactory<components>::value>,
          public def::Time,                                           // this handles time
          public def::NoNumFlux<ModelConstant<dim, components>>,      // use no numflux
          public def::FlowBoundaries<ModelConstant<dim, components>>, // use Inflow/Outflow boundaries
          public def::AD<ModelConstant<dim, components>>              // define all jacobians per AD
    {
    public:
      const PhysicalParameters prm;

      ModelConstant(PhysicalParameters prm) : prm(prm) {}
      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        for (uint c = 0; c < components; ++c)
          values[c] = prm.initial_x0[c] + prm.initial_x1[c] * pos[c];
      }

      template <typename Vector> std::array<double, dim> EoM(const Point<dim> &x, const Vector &u) const
      {
        // Just to avoid warnings
        (void)x;
        if constexpr (dim == 1)
          return std::array<double, dim>{{u[0]}};
        else if constexpr (dim == 2) {
          return std::array<double, dim>{{u[0], u[1]}};
        } else if constexpr (dim == 3)
          return std::array<double, dim>{{u[0], u[1], u[2]}};
        else
          throw std::runtime_error("Only 1, 2, and 3 dimensions are supported.");
      }

      double solution(const Point<dim> &pos) const { return prm.initial_x0[0] + prm.initial_x1[0] * pos[0]; }
    };

    template <uint dim>
    class LDGModelConstant
        : public def::AbstractModel<LDGModelConstant<dim>,
                                    ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<>,
                                                        ExtractorDescriptor<>, FEFunctionDescriptor<Scalar<"l">>>>,
          public def::Time,                                  // this handles time
          public def::NoNumFlux<LDGModelConstant<dim>>,      // use no numflux
          public def::FlowBoundaries<LDGModelConstant<dim>>, // use Inflow/Outflow boundaries
          public def::AD<LDGModelConstant<dim>>              // define all jacobians per AD
    {
    public:
      const PhysicalParameters prm;

      LDGModelConstant(PhysicalParameters prm) : prm(prm) {}
      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }
      double solution(const Point<dim> &pos) const { return prm.initial_x0[0] + prm.initial_x1[0] * pos[0]; }
      template <uint dependent, int mdim, typename NumberType, typename Solutions_s, typename Solutions_n>
      void ldg_numflux(std::array<Tensor<1, mdim, NumberType>, 1> &, const Tensor<1, mdim> &, const Point<mdim> &,
                       const Solutions_s &, const Solutions_n &) const
      {
      }
    };

    template <uint dim>
    class ModelExp : public def::AbstractModel<ModelExp<dim>, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
                     public def::Time,                          // this handles time
                     public def::NoNumFlux<ModelExp<dim>>,      // use no numflux
                     public def::FlowBoundaries<ModelExp<dim>>, // use Inflow/Outflow boundaries
                     public def::AD<ModelExp<dim>>              // define all jacobians per AD
    {
    protected:
      const PhysicalParameters prm;

    public:
      ModelExp(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      double solution(const Point<dim> &pos) const
      {
        return std::exp(t) * (prm.initial_x0[0] + prm.initial_x1[0] * pos[0]);
      }

      template <typename NT, typename Solution>
      void source(std::array<NT, 1> &s_i, const Point<dim> & /*p*/, const Solution &sol) const
      {
        const auto &fe_functions = std::get<0>(sol);
        s_i[0] = -fe_functions[0];
      }
    };

    template <uint dim>
    class ModelBurgers
        : public def::AbstractModel<ModelBurgers<dim>, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
          public def::Time,                              // this handles time
          public def::LLFFlux<ModelBurgers<dim>>,        // use LL numflux
          public def::FlowBoundaries<ModelBurgers<dim>>, // use Inflow/Outflow boundaries
          public def::AD<ModelBurgers<dim>>              // define all jacobians per AD
    {
    protected:
      const PhysicalParameters prm;

    public:
      ModelBurgers(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      double solution(const Point<dim> &pos) const
      {
        return (prm.initial_x0[0] + prm.initial_x1[0] * pos[0]) / (prm.initial_x1[0] * t + 1.);
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_functions = std::get<0>(sol);
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
      }
    };
  } // namespace Testing
} // namespace DiFfRG