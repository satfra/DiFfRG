#pragma once

// external libraries
#include <array>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>
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
      // The flux and source read only the values: the batches skip derivatives and hessians.
      static constexpr bool batch_reads_derivatives = false;
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

      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        // The return type is spelled out because nvcc's cudafe strips the braces from a
        // single-element `return {...}` in a class template member, leaving g++ with a bare double.
        return std::array<double, 1>{prm.initial_x0[0] + prm.initial_x1[0] * pos[0]};
      }
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
      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        return prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }
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
      // The flux and source read only the values: the batches skip derivatives and hessians.
      static constexpr bool batch_reads_derivatives = false;
      ModelExp(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        // Explicit return type: see the note in ModelConstant::solution.
        return std::array<double, 1>{std::exp(t) * (prm.initial_x0[0] + prm.initial_x1[0] * pos[0])};
      }

      template <typename NT, typename Solution>
      void source(std::array<NT, 1> &s_i, const Point<dim> & /*p*/, const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
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
      // The flux and source read only the values: the batches skip derivatives and hessians.
      static constexpr bool batch_reads_derivatives = false;
      ModelBurgers(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        // Explicit return type: see the note in ModelConstant::solution.
        return std::array<double, 1>{(prm.initial_x0[0] + prm.initial_x1[0] * pos[0]) / (prm.initial_x1[0] * t + 1.)};
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
      }
    };

    // A coupled FEM + explicit-variable model used to exercise the hybrid IDA + explicit
    // timesteppers (TimeStepperSUNDIALS_IDA_Boost{ABM,RK}).
    //
    // The FEM function u obeys the logistic ODE  du/dt = K u (1 - u)  at every spatial
    // point, i.e. u(t) = 1 / (1 + exp(-K (t - t_mid))).  For a large K this is a sharp
    // transition centred at t_mid: u stays near 0 for a long time and then rises steeply
    // to 1.  The implicit IDA controller grows its step in the flat region and is then
    // forced to reject and retry steps as it crosses the transition.
    //
    // The explicit variable v obeys  dv/dt = -v u(t)^2 , i.e. it is coupled to the FEM
    // solution.  Using  u^2 = u - (1/K) du/dt  and  \int u dt = (1/K) ln(1 + e^{K(t-t_mid)}),
    // its exact solution is
    //     v(t) = v0 exp( -[ Iu(t) - Iu(0) - (u(t) - u(0)) / K ] ),
    //     Iu(t) = (1/K) ln(1 + e^{K (t - t_mid)}).
    //
    // If a rejected IDA step is allowed to leak into the explicit-variable buffer, the
    // variable is integrated along a trajectory that IDA never accepted and v(t_final)
    // deviates from the exact value above.  That is the regression this model guards
    // against.
    template <uint dim>
    class ModelHybridRollback
        : public def::AbstractModel<
              ModelHybridRollback<dim>,
              ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<Scalar<"v">>,
                                  ExtractorDescriptor<Scalar<"u_eom">>>>,
          public def::Time,                                // this handles time
          public def::NoNumFlux<ModelHybridRollback<dim>>, // pure source term, no spatial coupling
          public def::FlowBoundaries<ModelHybridRollback<dim>>,
          public def::AD<ModelHybridRollback<dim>> // define all jacobians per AD
    {
    public:
      // The flux and source read only the values: the batches skip derivatives and hessians.
      static constexpr bool batch_reads_derivatives = false;
      static constexpr double K = 30.;     // sharpness of the FEM transition
      static constexpr double t_mid = 0.5; // location of the FEM transition
      static constexpr double v0 = 1.;     // initial value of the explicit variable

      const PhysicalParameters prm;
      ModelHybridRollback(PhysicalParameters prm) : prm(prm) {}

      static double u_exact(double tt) { return 1. / (1. + std::exp(-K * (tt - t_mid))); }
      // antiderivative of u_exact
      static double Iu(double tt) { return std::log1p(std::exp(K * (tt - t_mid))) / K; }

      template <typename Vector> void initial_condition(const Point<dim> & /*pos*/, Vector &values) const
      {
        values[0] = u_exact(0.);
      }
      template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = v0; }

      // du/dt = K u (1 - u);  the residual convention is u_dot = -source.
      template <typename NT, typename Solution>
      void source(std::array<NT, 1> &s_i, const Point<dim> & /*p*/, const Solution &sol) const
      {
        const auto u = get<0>(sol)[0];
        s_i[0] = -K * u * (1. - u);
      }

      // u is spatially constant, so the extraction location is irrelevant; a fixed, well
      // defined root at x = 0.5 simply keeps the EoM search well-posed.
      template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector & /*u*/) const
      {
        return {{x[0] - 0.5}};
      }

      // hand the FEM solution at the EoM point to the explicit-variable residual
      template <typename NT, typename Solution>
      void extract(std::array<NT, 1> &extractors, const Point<dim> & /*x*/, const Solution &sol) const
      {
        extractors[0] = get<"fe_functions">(sol)[0];
      }

      // dv/dt = -v u^2;  the residual convention is v_dot = -r_a.
      template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &sol) const
      {
        const auto u = get<"extractors">(sol)[0];
        const auto v = get<"variables">(sol)[0];
        r_a[0] = v * u * u;
      }

      // exact FEM solution (validated by the standard test harness, block 0)
      double solution(const Point<dim> & /*pos*/) const { return u_exact(t); }
      // exact explicit-variable solution (validated by the dedicated test, block 1)
      double variable_solution() const
      {
        // \int_0^t u^2 dt' = (Iu(t) - Iu(0)) - (u(t) - u(0)) / K
        const double int_u2 = (Iu(t) - Iu(0.)) - (u_exact(t) - u_exact(0.)) / K;
        return v0 * std::exp(-int_u2);
      }
    };

    // A smooth coupled FEM + explicit-variable model used to measure how the hybrid
    // timesteppers behave when each IDA trial step contains only one or two explicit
    // sub-steps -- the regime in which a typical FRG production run operates and the
    // regime in which a per-trial-step reset() of the multistep stepper would be most
    // visible.  K is chosen small enough that IDA does not need to reject any steps,
    // so the rollback machinery is not exercised; the test is purely about whether the
    // explicit integration retains its design accuracy on accepted steps.
    //
    // FEM:        du/dt = -K u                ->  u(t) = u0 exp(-K t)
    // variable:   dv/dt = -u                  ->  v(t) = v0 + (u0/K) (exp(-K t) - 1)
    template <uint dim>
    class ModelHybridSmooth
        : public def::AbstractModel<ModelHybridSmooth<dim>, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>,
                                                                                VariableDescriptor<Scalar<"v">>,
                                                                                ExtractorDescriptor<Scalar<"u_eom">>>>,
          public def::Time, // this handles time
          public def::NoNumFlux<ModelHybridSmooth<dim>>,
          public def::FlowBoundaries<ModelHybridSmooth<dim>>,
          public def::AD<ModelHybridSmooth<dim>>
    {
    public:
      // The flux and source read only the values: the batches skip derivatives and hessians.
      static constexpr bool batch_reads_derivatives = false;
      static constexpr double K = 2.;
      static constexpr double u0 = 1.;
      static constexpr double v0 = 0.;

      const PhysicalParameters prm;
      ModelHybridSmooth(PhysicalParameters prm) : prm(prm) {}

      static double u_exact(double tt) { return u0 * std::exp(-K * tt); }

      template <typename Vector> void initial_condition(const Point<dim> & /*pos*/, Vector &values) const
      {
        values[0] = u0;
      }
      template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = v0; }

      // du/dt = -K u;  residual convention is u_dot = -source.
      template <typename NT, typename Solution>
      void source(std::array<NT, 1> &s_i, const Point<dim> & /*p*/, const Solution &sol) const
      {
        const auto u = get<0>(sol)[0];
        s_i[0] = K * u;
      }

      template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector & /*u*/) const
      {
        return {{x[0] - 0.5}};
      }

      template <typename NT, typename Solution>
      void extract(std::array<NT, 1> &extractors, const Point<dim> & /*x*/, const Solution &sol) const
      {
        extractors[0] = get<"fe_functions">(sol)[0];
      }

      // dv/dt = -u;  residual convention is v_dot = -r_a.
      template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &sol) const
      {
        const auto u = get<"extractors">(sol)[0];
        r_a[0] = u;
      }

      double solution(const Point<dim> & /*pos*/) const { return u_exact(t); }
      double variable_solution() const { return v0 + (u0 / K) * (std::exp(-K * t) - 1.); }
    };

    // A coupled FEM + explicit-variable model with two-way coupling, as in a production
    // FRG model where the explicit-variable wave-function renormalisations feed back into
    // the FEM source: the FEM equation reads v through the extractor and v's residual
    // reads u.  The ModelHybridRollback test only has one-way coupling (FEM drives v but
    // v does not feed back into FEM), so a regression in how the explicit variable is
    // served back to IDA during Newton iterations cannot be detected by it.
    //
    // FEM:        du/dt = -K (u - v)         (FEM source reads the variable v via extractor[1])
    // variable:   dv/dt = +K (u - v)         (variable residual reads u via extractor[0])
    //
    // Conserved:  u + v = u0 + v0
    // Decaying :  u - v = (u0 - v0) exp(-2 K t)
    //   =>  u(t) = (u0+v0)/2 + (u0-v0)/2 exp(-2 K t)
    //       v(t) = (u0+v0)/2 - (u0-v0)/2 exp(-2 K t)
    template <uint dim>
    class ModelHybridTwoWay
        : public def::AbstractModel<
              ModelHybridTwoWay<dim>,
              ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<Scalar<"v">>,
                                  ExtractorDescriptor<Scalar<"u_eom">, Scalar<"v_eom">>>>,
          public def::Time, // this handles time
          public def::NoNumFlux<ModelHybridTwoWay<dim>>,
          public def::FlowBoundaries<ModelHybridTwoWay<dim>>,
          public def::AD<ModelHybridTwoWay<dim>>
    {
    public:
      // The flux and source read only the values: the batches skip derivatives and hessians.
      static constexpr bool batch_reads_derivatives = false;
      static constexpr double K = 5.;
      static constexpr double u0 = 1.;
      static constexpr double v0 = 0.;

      const PhysicalParameters prm;
      ModelHybridTwoWay(PhysicalParameters prm) : prm(prm) {}

      static double u_exact(double tt) { return 0.5 * (u0 + v0) + 0.5 * (u0 - v0) * std::exp(-2. * K * tt); }
      static double v_exact(double tt) { return 0.5 * (u0 + v0) - 0.5 * (u0 - v0) * std::exp(-2. * K * tt); }

      template <typename Vector> void initial_condition(const Point<dim> & /*pos*/, Vector &values) const
      {
        values[0] = u0;
      }
      template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = v0; }

      // du/dt = -K (u - v);  residual convention is u_dot = -source, source = K (u - v).
      template <typename NT, typename Solution>
      void source(std::array<NT, 1> &s_i, const Point<dim> & /*p*/, const Solution &sol) const
      {
        const auto u = get<0>(sol)[0];
        const auto v = get<"extractors">(sol)[1];
        s_i[0] = K * (u - v);
      }

      template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector & /*u*/) const
      {
        return {{x[0] - 0.5}};
      }

      template <typename NT, typename Solution>
      void extract(std::array<NT, 2> &extractors, const Point<dim> & /*x*/, const Solution &sol) const
      {
        extractors[0] = get<"fe_functions">(sol)[0];
        extractors[1] = get<"variables">(sol)[0];
      }

      // dv/dt = +K (u - v);  residual convention is v_dot = -r_a.
      template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &sol) const
      {
        const auto u = get<"extractors">(sol)[0];
        const auto v = get<"variables">(sol)[0];
        r_a[0] = -K * (u - v);
      }

      double solution(const Point<dim> & /*pos*/) const { return u_exact(t); }
      double variable_solution() const { return v_exact(t); }
    };

    // A two-way coupled model in which the EXPLICIT variable only WEAKLY depends on the
    // implicit (spatial) block: the FEM source still reads v at full strength (so serving v
    // back to IDA matters), but v's own residual reads u scaled by a small eps. This mimics
    // a regime where the expensive explicit sector is nearly autonomous w.r.t. the cheap
    // implicit one. With weak arrow-1 coupling, the explicit trajectory is robust to small
    // errors in the spatial data fed into it -- the failure mode that destabilises the
    // single-buffer stagger (mode 1) -- so this isolates how the serving choice (predictor
    // vs lookahead) affects the *implicit* accuracy.
    //
    // FEM:        du/dt = -K (u - v)            (source reads v via extractor[1], full K)
    // variable:   dv/dt = +eps K (u - v)        (residual reads u via extractor[0], weak)
    //
    // With w = u - v:  dw/dt = -K(1+eps) w  =>  w(t) = w0 exp(-K(1+eps) t),  w0 = u0 - v0.
    //   u(t) = u0 - (w0/(1+eps)) (1 - exp(-K(1+eps) t))
    //   v(t) = v0 + (eps w0/(1+eps)) (1 - exp(-K(1+eps) t))
    template <uint dim>
    class ModelHybridWeakCoupling
        : public def::AbstractModel<
              ModelHybridWeakCoupling<dim>,
              ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<Scalar<"v">>,
                                  ExtractorDescriptor<Scalar<"u_eom">, Scalar<"v_eom">>>>,
          public def::Time, // this handles time
          public def::NoNumFlux<ModelHybridWeakCoupling<dim>>,
          public def::FlowBoundaries<ModelHybridWeakCoupling<dim>>,
          public def::AD<ModelHybridWeakCoupling<dim>>
    {
    public:
      // The flux and source read only the values: the batches skip derivatives and hessians.
      static constexpr bool batch_reads_derivatives = false;
      static constexpr double K = 5.;
      static constexpr double eps = 0.1; // weak dependence of the explicit variable on u
      static constexpr double u0 = 1.;
      static constexpr double v0 = 0.;

      const PhysicalParameters prm;
      ModelHybridWeakCoupling(PhysicalParameters prm) : prm(prm) {}

      static double g(double tt) { return 1. - std::exp(-K * (1. + eps) * tt); }
      static double u_exact(double tt) { return u0 - ((u0 - v0) / (1. + eps)) * g(tt); }
      static double v_exact(double tt) { return v0 + (eps * (u0 - v0) / (1. + eps)) * g(tt); }

      template <typename Vector> void initial_condition(const Point<dim> & /*pos*/, Vector &values) const
      {
        values[0] = u0;
      }
      template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = v0; }

      // du/dt = -K (u - v);  residual convention is u_dot = -source, source = K (u - v).
      template <typename NT, typename Solution>
      void source(std::array<NT, 1> &s_i, const Point<dim> & /*p*/, const Solution &sol) const
      {
        const auto u = get<0>(sol)[0];
        const auto v = get<"extractors">(sol)[1];
        s_i[0] = K * (u - v);
      }

      template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector & /*u*/) const
      {
        return {{x[0] - 0.5}};
      }

      template <typename NT, typename Solution>
      void extract(std::array<NT, 2> &extractors, const Point<dim> & /*x*/, const Solution &sol) const
      {
        extractors[0] = get<"fe_functions">(sol)[0];
        extractors[1] = get<"variables">(sol)[0];
      }

      // dv/dt = +eps K (u - v);  residual convention is v_dot = -r_a.
      template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &sol) const
      {
        const auto u = get<"extractors">(sol)[0];
        const auto v = get<"variables">(sol)[0];
        r_a[0] = -eps * K * (u - v);
      }

      double solution(const Point<dim> & /*pos*/) const { return u_exact(t); }
      double variable_solution() const { return v_exact(t); }
    };
  } // namespace Testing
} // namespace DiFfRG
