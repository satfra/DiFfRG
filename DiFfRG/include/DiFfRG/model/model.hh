#pragma once

// external libraries
#include "DiFfRG/common/config_tree.hh"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// standard library
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

// DiFfRG
#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/model/abs_tolerances.hh>
#include <DiFfRG/model/ad.hh>
#include <DiFfRG/model/component_descriptor.hh>
#include <DiFfRG/model/fv_boundaries.hh>
#include <DiFfRG/model/numflux.hh>

// clang-format off
/**
 * @file model.hh
 * @brief def::AbstractModel, the base class of every user model, and the default building blocks in def::.
 *
 * A model is a class `MyModel : public def::AbstractModel<MyModel, Components>, ...` that overrides the hooks
 * below; everything it does not override keeps the default. A guided introduction is in
 * documentation/getting_started/models.md, batched evaluation in documentation/tutorials/tut6.md.
 *
 * ### What is solved
 * The unknowns are declared in a ComponentDescriptor (component_descriptor.hh):
 * - FE functions \f$u_i(x)\f$, \f$i = 0, \dots, N_f - 1\f$, fields on the discretized field space
 *   \f$x \in \mathbb{R}^d\f$ (\f$d\f$ = `dim`),
 * - variables \f$v_a\f$, \f$a = 0, \dots, N_v - 1\f$, extra unknowns without \f$x\f$ dependence,
 * - extractors \f$e_b\f$, \f$b = 0, \dots, N_e - 1\f$: not unknowns, but numbers computed from the \f$u_i\f$ at a
 *   single point (by default the EoM point, see EoM()) by extract().
 *
 * With \f$\partial u\f$, \f$\partial^2 u\f$ the gradient and hessian in \f$x\f$, every PDE assembler (CG, DG, LDG,
 * FV/Kurganov-Tadmor) solves
 * \f[
 *   m_i(u, \partial_t u, x) + \nabla\cdot F_i(u, \partial u, \partial^2 u, e, v, x)
 *     + s_i(u, \partial u, \partial^2 u, e, v, x) = 0\,,
 *   \qquad
 *   \partial_t v_a = -D_a(v, e)\,,
 * \f]
 * with mass \f$m\f$ = mass(), flux \f$F\f$ = flux(), source \f$s\f$ = source() and \f$D\f$ = dt_variables().
 * With the default mass \f$m_i = \partial_t u_i\f$ this is \f$\partial_t u_i = -\nabla\cdot F_i - s_i\f$: the source
 * enters with a plus sign on the left-hand side.
 *
 * ### Hooks
 * | Hook | What it computes | Used by |
 * |------|------------------|---------|
 * | initial_condition | \f$u_i(x)\f$ at \f$t = 0\f$ | all PDE assemblers |
 * | mass | \f$m_i\f$, default \f$\partial_t u_i\f$ | CG, DG, LDG, KT |
 * | flux | \f$F_i\f$ | CG, DG, LDG: full flux; KT: advective part |
 * | diffusion_flux | parabolic part of \f$F_i\f$, added to flux at faces | KT |
 * | source | \f$s_i\f$ | all PDE assemblers |
 * | evaluate_batch | flux, source, diffusion_flux at many points at once | all PDE assemblers |
 * | differential_components | which \f$m_i\f$ depend on \f$\partial_t u\f$ | IDA timesteppers |
 * | wave_speed_blocks | which components share a dissipation speed | KT |
 * | initial_condition_variables | \f$v_a\f$ at \f$t = 0\f$ | all |
 * | dt_variables | \f$D_a = -\partial_t v_a\f$ | all |
 * | EoM, EoM_postprocess | the EoM point, where extractors and readouts are evaluated | all PDE assemblers |
 * | extract | \f$e_b\f$ | all PDE assemblers |
 * | raw_potential_gradient | gradient of the potential reconstructed for extract, readouts | all PDE assemblers |
 * | readouts_multiple, readouts | output at the EoM point(s) | all |
 * | face_indicator, cell_indicator | mesh refinement indicators | DG, LDG: faces, cells; CG: cells |
 * | affine_constraints | constraints on individual dofs | all PDE assemblers |
 * | ldg_flux, ldg_source, ldg_evaluate_batch | the LDG levels, see below | LDG |
 * | `extractor_point` (optional) | extractor point from the whole solution, see EoM_postprocess | all PDE assemblers |
 * | `abs_tolerances` (optional) | per-dof absolute tolerance, see abs_tolerances.hh | LDG, KT |
 *
 * ### Traits
 * Optional `static constexpr bool` members of the model:
 * | Trait | Default | Effect |
 * |-------|---------|--------|
 * | `batch_reads_derivatives` | true | CG, DG: false = no term reads \f$\partial u\f$, so it is not gathered |
 * | `batch_reads_hessians` | true | CG, DG: false = no term reads \f$\partial^2 u\f$, so it is not gathered |
 * | `source_uses_hessians` | false | KT: true = source() gets `"fe_hessians"` (diagonal curvature) |
 * | `extract_uses_potential` | true | false = extract() ignores the potential; saves a global solve per evaluation |
 * | `readouts_use_potential` | true | false = potential reconstructed for readouts only if the output writes it |
 *
 * ### Building blocks in def::
 * A model inherits, besides AbstractModel, one class of each row it needs (pass the model as template argument
 * where there is one):
 * | Purpose | Classes |
 * |---------|---------|
 * | RG time | def::fRG (\f$k = \Lambda e^{-t}\f$; members `t`, `k`, `k2`...`k6`) or def::Time (`t`); not both |
 * | Jacobians | def::AD, def::FE_AD, def::NoJacobians; see ad.hh |
 * | Interior numerical flux | def::LLFFlux, def::NoNumFlux (DG, LDG main level); def::LDGUpDownFluxes (LDG levels) |
 * | Boundary numerical flux | def::FlowBoundaries (CG, DG, LDG) |
 * | Boundary ghost cells (KT) | def::FVDefaultBoundaries and the other mixins in fv_boundaries.hh |
 * | Affine constraints | def::ConstrainOrigin{Boundary,Support}PointToZero<"u", Model>, def::NoAffineConstraints |
 *
 * ### LDG levels
 * The LDG assembler constructs up to three auxiliary levels \f$L_1, L_2, L_3\f$ from \f$L_0 = u\f$:
 * \f[
 *   L_k = \nabla\cdot F^{(k)}(L_{k-1}, x) + S^{(k)}(L_{k-1}, x)\,, \qquad k = 1, 2, 3\,,
 * \f]
 * with \f$F^{(k)}\f$ = ldg_flux<k> and \f$S^{(k)}\f$ = ldg_source<k>. There is no minus sign: in 1D, `ldg_flux<1>`
 * returning \f$u\f$ gives \f$L_1 = +\partial_x u\f$. The main equation reads \f$L_1, L_2, L_3\f$ from its solution
 * tuple as `"LDG1"`, `"LDG2"`, `"LDG3"`.
 *
 * In the model's constructor, `components().add_dependency(k, i, k - 1, j)` declares that component i of level k
 * depends on component j of level k - 1. Only adjacent levels are declared; couplings across several levels are
 * composed automatically, and the order of the calls does not matter. `components().set_jacobian_constant(k, k - 1)`
 * declares level k a fixed linear, homogeneous map of level k - 1 (no constant terms anywhere, including boundary
 * fluxes and sources); the assembler then builds that map once, provided all levels up to k are declared constant.
 * The declaration is checked on first use and throws if violated.
 */
// clang-format on

namespace DiFfRG
{
  using namespace dealii;

  /// Default implementations and building blocks for user models, see model.hh.
  namespace def
  {
    /// Block id for a component that carries no hyperbolic flux. @see AbstractModel::wave_speed_blocks.
    inline constexpr int no_wave_speed = -1;

    /**
     * @brief Base class of every model: declares all hooks an assembler may call, with defaults.
     *
     * Derive as `class MyModel : public def::AbstractModel<MyModel, Components>` and shadow the hooks you need; the
     * signatures below are what the assemblers call. Template parameters such as the number type can be chosen
     * freely as long as the call still resolves: the same hook is called with `double` and with AD types. See
     * model.hh for the equation and an overview of all hooks.
     *
     * Most hooks receive the solution at a point as a named tuple `sol`; read its entries by name, e.g.
     * `get<"fe_functions">(sol)`. Which entries exist depends on the assembler (see the hooks); reading one that
     * does not exist is a compile error.
     *
     * @tparam Model the derived model class (CRTP).
     * @tparam Components_ a DiFfRG::ComponentDescriptor.
     */
    template <typename Model, typename Components_> class AbstractModel
    {
      Model &asImp()
      {
        static_assert(
            std::is_base_of_v<AbstractModel<Model, Components_>, Model>,
            "AbstractModel<Model, Components>: Model must inherit from AbstractModel<Model, Components> (CRTP). "
            "Check that your model class passes itself as the first template argument.");
        return static_cast<Model &>(*this);
      }
      const Model &asImp() const
      {
        static_assert(
            std::is_base_of_v<AbstractModel<Model, Components_>, Model>,
            "AbstractModel<Model, Components>: Model must inherit from AbstractModel<Model, Components> (CRTP). "
            "Check that your model class passes itself as the first template argument.");
        return static_cast<const Model &>(*this);
      }

    protected:
      Components_ m_components;
      /// The component descriptor, writable: declare LDG dependencies here in the model's constructor.
      auto &components() { return m_components; }

    public:
      /// The component descriptor.
      const auto &get_components() const { return m_components; }
      using Components = Components_;

      /**
       * @name Equation
       * The field-space PDE \f$m + \nabla\cdot F + s = 0\f$, see model.hh.
       */
      //@{

      /**
       * @brief Initial condition of the FE functions. Must be provided by any model with FE functions.
       *
       * @param x the point in field space.
       * @param u_i to be filled with \f$u_i(x)\f$ at \f$t = 0\f$, one entry per FE function.
       */
      template <int dim, typename Vector> void initial_condition(const Point<dim> &x, Vector &u_i) const = delete;

      /**
       * @brief The mass term \f$m_i(u, \partial_t u, x)\f$. Default: \f$m_i = \partial_t u_i\f$.
       *
       * Override for algebraic components (\f$m_i\f$ independent of \f$\partial_t u\f$, e.g. \f$m_i = 0\f$, or
       * \f$m_i = u_i\f$ for a constraint \f$u_i + \dots = 0\f$) or a non-trivial time-derivative structure. Explicit
       * timesteppers need \f$m\f$ linear in \f$\partial_t u\f$; algebraic components need IDA.
       *
       * @param m_i the result, one entry per FE function.
       * @param x the point in field space.
       * @param u_i the values \f$u_i(x)\f$.
       * @param dt_u_i the time derivatives \f$\partial_t u_i(x)\f$.
       */
      template <int dim, typename NumberType, typename Vector, typename Vector_dot, size_t n_fe_functions>
      void mass([[maybe_unused]] std::array<NumberType, n_fe_functions> &m_i, [[maybe_unused]] const Point<dim> &x,
                [[maybe_unused]] const Vector &u_i, const Vector_dot &dt_u_i) const
      {
        for (uint i = 0; i < n_fe_functions; ++i)
          m_i[i] = dt_u_i[i];
      }

      /**
       * @brief The flux \f$F_i(u, \partial u, \partial^2 u, e, v, x)\f$ at one point. Default: zero.
       *
       * CG, DG and LDG: the full flux of the equation. KT: only the advective (hyperbolic) part, from which the
       * numerical face flux and wave speeds are built; the parabolic part goes into diffusion_flux(). KT evaluates
       * this separately on both traces of a face, with `"fe_derivatives"` the gradients of the advection
       * reconstruction.
       *
       * @param F_i the result, one \f$d\f$-vector per FE function.
       * @param x the point in field space.
       * @param sol the solution at @p x; for its entries see source().
       *
       * @note The extractors in @p sol are those of the current state. Whether the jacobian contains their
       * dependence on \f$u\f$: yes under def::AD on CG, DG and LDG; no under def::FE_AD and always no on KT (the
       * extractors are frozen within a Newton step). A frozen dependence costs Newton iterations, not accuracy.
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief KT only: the diffusive (parabolic) part \f$D_i\f$ of the flux at a face. Default: zero.
       *
       * The face flux is \f$(H + D_i)\cdot n\f$, where \f$H\f$ is the numerical flux built from flux(): both are
       * physical fluxes with the same sign convention, and they are added. Forward diffusion therefore needs a
       * flux decreasing in the gradient, e.g. \f$D = -\nu\,\partial_x u\f$. KT evaluates this separately on both
       * traces of a face, with gradients from the diffusion reconstruction (not those handed to flux()).
       *
       * @param F_i the result, one \f$d\f$-vector per FE function.
       * @param x the point in field space.
       * @param sol named tuple: `"fe_functions"`, `"fe_derivatives"`, `"fe_third_derivatives"` (\f$\partial_x^3
       * u_i\f$, 1D only; zero otherwise), `"extractors"`, `"variables"`, `"cell_width"`. Extractors are frozen, as
       * for flux().
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void diffusion_flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                          [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief The source \f$s_i(u, \partial u, \partial^2 u, e, v, x)\f$ at one point. Default: zero.
       *
       * Note the sign: \f$s\f$ is on the left-hand side, \f$\partial_t u_i = -\nabla\cdot F_i - s_i\f$.
       *
       * @param s_i the result, one entry per FE function.
       * @param x the point in field space.
       * @param sol named tuple; its entries depend on the assembler:
       * - CG, DG: `"fe_functions"` \f$u_i\f$, `"fe_derivatives"` \f$\partial u_i\f$ (unless
       *   `batch_reads_derivatives = false`), `"fe_hessians"` \f$\partial^2 u_i\f$ (unless `batch_reads_hessians =
       *   false`), `"extractors"` \f$e_b\f$, `"variables"` \f$v_a\f$, `"cell_width"`.
       * - LDG main level: `"fe_functions"`, `"LDG1"`, ... (the LDG levels), `"extractors"`, `"variables"`,
       *   `"cell_width"`.
       * - KT: `"fe_functions"`, `"fe_derivatives"` (the limited reconstructed gradient), `"fe_hessians"` only with
       *   `source_uses_hessians = true`, `"extractors"`, `"variables"`, `"cell_width"`.
       *
       * The same tuple is passed to flux() (on KT faces without hessians).
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void source([[maybe_unused]] std::array<NumberType, n_fe_functions> &s_i, [[maybe_unused]] const Point<dim> &x,
                  [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief Flux, source and/or diffusion flux at all points of a batch at once; all PDE assemblers evaluate the
       * model through this. See documentation/tutorials/tut6.md.
       *
       * The default calls flux(), source() and diffusion_flux() point by point, in parallel (so they must be
       * thread safe). Override it to evaluate the expensive part, typically the momentum integrals, for all points
       * in one call to an integrator's map_points(), which also runs on the GPU. The AD jacobians of def::AD and
       * def::FE_AD differentiate this function, and def::LLFFlux / def::FlowBoundaries evaluate face fluxes through
       * it.
       *
       * @param out a BatchOutput: write `out.flux(c, d)[i]`, `out.source(c)[i]`, `out.diffusion_flux(c, d)[i]` for
       * component c, direction d, point i. Compute only what `out.requested(Term::...)` asks for; all columns are
       * zero on entry.
       * @param batch a PointBatch: `values(c)`, `derivatives(c, d)`, `hessians(c, d1, d2)`, `coordinates(d)`,
       * `extractors()`, `variables()`, `x(i)`, `cell_width(i)`, `size()`. LDG adds `ldg_values(k, c)`, KT diffusion
       * traces `third_derivatives(c, d0, d1, d2)`. Inputs the assembler does not gather do not compile; test with
       * `if constexpr (Batch::has_derivatives)` / `Batch::has_hessians`.
       *
       * @warning `typename Batch::number_type` may be double or an AD type, and the batch may hold several stacked
       * copies of the assembler's points (AD seeds, LLF perturbations): size everything from `batch.size()` and
       * keep no per-point state between calls.
       */
      template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
      {
        DiFfRG::internal::evaluate_per_point(asImp(), out, batch);
      }

      /**
       * @brief Which FE components are differential (true) or algebraic (false), for the IDA timesteppers.
       *
       * The default perturbs each \f$\partial_t u_j\f$ in mass() at \f$u = \partial_t u = 1\f$, \f$x = (1, \dots,
       * 1)\f$ and marks every \f$m_i\f$ that changes. Override if that test point is not representative.
       */
      template <uint dim> std::vector<bool> differential_components() const
      {
        std::vector<bool> differential_components(Model::Components::count_fe_functions(), false);

        // First we need two reference solutions u_i and dt_u_i, which we then both fill with 1.s
        std::array<double, Model::Components::count_fe_functions()> u_i{{}};
        std::array<double, Model::Components::count_fe_functions()> dt_u_i{{}};
        for (uint i = 0; i < Model::Components::count_fe_functions(); ++i) {
          u_i[i] = 1.;
          dt_u_i[i] = 1.;
        }
        // Set the point to be at 1. in all directions
        Point<dim> x;
        for (uint i = 0; i < dim; ++i)
          x[i] = 1.;
        // Get the mass function m_i
        std::array<double, Model::Components::count_fe_functions()> m_i{{}};
        asImp().mass(m_i, x, u_i, dt_u_i);

        // Now we check which components are differential by changing dt_u_i slightly and checking whether the mass
        // function changes.
        for (uint i = 0; i < Model::Components::count_fe_functions(); ++i) {
          dt_u_i[i] = 1. + 1e-1;
          std::array<double, Model::Components::count_fe_functions()> m_i_new{{}};
          asImp().mass(m_i_new, x, u_i, dt_u_i);
          dt_u_i[i] = 1.;
          for (uint j = 0; j < Model::Components::count_fe_functions(); ++j)
            if (!is_close(m_i[j], m_i_new[j])) differential_components[j] = true;
        }

        return differential_components;
      }

      /**
       * @brief KT only: which FE components share a wave speed, and which carry no hyperbolic flux.
       *
       * KT adds a dissipation \f$-\tfrac{a}{2}(u^+ - u^-)\f$ to the numerical flux of every component, with \f$a\f$
       * the spectral radius of the flux jacobian \f$\partial F/\partial u\f$. Components with the same block id
       * share one \f$a\f$, computed from the jacobian restricted to that block. A component marked
       * def::no_wave_speed gets no dissipation and a zero numerical flux, so its flux() must be zero.
       * Default: all components in block 0.
       *
       * Use separate blocks when the system mixes a conservation law with algebraic constraints:
       * - an algebraic component without flux should be `no_wave_speed`; otherwise the dissipation dominates its
       *   row and pulls the slope limiter into the Newton jacobian.
       * - a component with a much slower flux than the fastest one is over-dissipated in a shared block; if its row
       *   is algebraic, the row loses diagonal dominance as the mesh is refined.
       *
       * Do not split a genuinely hyperbolic system: blocks assume the cross-couplings between them carry no
       * characteristics. The blocks must not depend on the solution.
       *
       * @param blocks one block id (>= 0) or def::no_wave_speed per FE function.
       */
      template <size_t n_fe_functions> void wave_speed_blocks(std::array<int, n_fe_functions> &blocks) const
      {
        blocks.fill(0);
      }

      //@}
      /**
       * @name Variables
       * Unknowns without field-space dependence, \f$\partial_t v_a = -D_a(v, e)\f$.
       */
      //@{

      /**
       * @brief Initial values of the variables. Default: leaves @p v_a untouched (zero).
       * @param v_a to be filled with \f$v_a\f$ at \f$t = 0\f$.
       */
      template <typename Vector> void initial_condition_variables([[maybe_unused]] Vector &v_a) const
      {
        // Just to avoid warnings
      }

      /**
       * @brief The time evolution of the variables, in residual form: fills \f$r_a = -\partial_t v_a\f$.
       *
       * @warning The sign: this returns \f$-\partial_t v_a\f$, not \f$\partial_t v_a\f$. A flow
       * \f$\partial_t \lambda = \beta(\lambda)\f$ is implemented as `r[idx("lambda")] = -beta(lambda)`. This holds
       * for all assemblers and timesteppers, including the variables-only assembler.
       *
       * @param r_a the result, one entry per variable.
       * @param sol named tuple: `"variables"` \f$v_a\f$ and `"extractors"` \f$e_b\f$ (evaluated at the extractor
       * point); the variables-only assembler passes only `"variables"`. Under def::AD the jacobian w.r.t. the
       * variables is the AD of this function.
       */
      template <typename Vector, typename Solution>
      void dt_variables([[maybe_unused]] Vector &r_a, [[maybe_unused]] const Solution &sol) const
      {
        // Just to avoid warnings
      }

      //@}
      /**
       * @name EoM, extractors and readouts
       * The EoM point \f$x_0\f$ is the root of EoM() (typically the minimum of the potential). The extractors are
       * read off the solution there (or at extractor_point, if defined), the readouts write output there.
       */
      //@{

      /**
       * @brief The function whose root is the EoM point: \f$x_0\f$ with \f$\mathrm{EoM}(x_0, u(x_0)) = 0\f$.
       * Default: the first FE function, \f$u_0\f$, in the first direction.
       *
       * @param x the point in field space.
       * @param u the values \f$u_i(x)\f$.
       */
      template <int dim, typename Vector>
      std::array<double, dim> EoM([[maybe_unused]] const Point<dim> &x, const Vector &u) const
      {
        return std::array<double, dim>{{u[0]}};
      }

      /**
       * @brief Move the EoM point after it has been found, from the point and the values there alone (e.g. to
       * freeze it once it jumps). Default: unchanged. Applied for the extractors and for the readouts.
       *
       * To choose the extractor point from the whole solution profile instead, define
       * `template <int dim, typename NT> Point<dim> extractor_point(const Point<dim> &EoM, const SolutionSample<dim,
       * NT> &sample) const` in the model (detected by DiFfRG::HasExtractorPoint; affects extract() only).
       */
      template <int dim, typename Vector> Point<dim> EoM_postprocess(const Point<dim> &EoM, const Vector &) const
      {
        return EoM;
      }

      /**
       * @brief Compute the extractors from the solution at a single point. Default: nothing.
       *
       * The results are what dt_variables(), flux() and source() see as `"extractors"`.
       *
       * @param result to be filled with \f$e_b\f$, one entry per extractor.
       * @param x the extractor point: the EoM point, or extractor_point() if the model defines it.
       * @param sol named tuple: `"fe_functions"`, `"fe_derivatives"`, `"fe_hessians"`, `"extractors"` (a placeholder,
       * do not read), `"variables"`, and the reconstructed scalar potential `"potential"`, `"potential_gradient"`,
       * `"potential_hessian"` (see raw_potential_gradient(); unusable if `extract_uses_potential = false`). LDG
       * adds `"LDG1"`, ... after `"fe_functions"`. Under def::AD the extractor jacobian is the AD of this function.
       */
      template <int dim, typename Vector, typename Solutions>
      void extract([[maybe_unused]] Vector &result, [[maybe_unused]] const Point<dim> &x,
                   [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief Whether extract() reads the potential slots of its tuple.
       *
       * Reconstructing the potential is a direct solve over the whole mesh on every residual and jacobian
       * evaluation. With false it is skipped and the slots hold a type without operations, so reading them is a
       * compile error.
       */
      static constexpr bool extract_uses_potential = true;

      /**
       * @brief Whether readouts() reads the potential slots of its tuple. With false, the potential is reconstructed
       * for readouts only if the VTK/HDF5 output writes it.
       */
      static constexpr bool readouts_use_potential = true;

      /**
       * @brief The gradient \f$\nabla_x U\f$ of the scalar potential \f$U(x)\f$ that is reconstructed for extract()
       * and readouts(). Default: the first `dim` FE functions (missing ones are zero).
       *
       * Kept separate from EoM(), which may contain terms (e.g. explicit symmetry breaking) that are not part of
       * the potential.
       */
      template <int dim, typename Vector>
      std::array<double, dim> raw_potential_gradient([[maybe_unused]] const Point<dim> &x, const Vector &u) const
      {
        std::array<double, dim> gradient{};
        for (uint d = 0; d < dim && d < u.size(); ++d)
          gradient[d] = u[d];
        return gradient;
      }

      /**
       * @brief Register the readouts, each with its own EoM. Default: one readout `"primary"` at the root of EoM(),
       * written by readouts().
       *
       * Override to output at several points: call `helper(id, eom_fun, out_fun)` once per readout, with
       * `eom_fun(x, u) -> std::array<double, dim>` (like EoM()) and `out_fun(output, x, sol)` (like readouts()).
       */
      template <typename FUN, typename DataOut> void readouts_multiple(FUN &helper, DataOut &) const
      {
        helper(
            "primary", [&](const auto &x, const auto &u_i) { return asImp().EoM(x, u_i); }, // chiral EoM
            [&](auto &output, const auto &x, const auto &sol) { asImp().readouts(output, x, sol); });
      }

      /**
       * @brief Write output at the EoM point, e.g. `output.table("data.csv").value("name", value)`. Default: none.
       *
       * @param output the output frame.
       * @param x the EoM point (after EoM_postprocess()).
       * @param sol as for extract(), with `"extractors"` holding the actual extractors.
       */
      template <int dim, typename DataOut, typename Solutions>
      void readouts([[maybe_unused]] DataOut &output, [[maybe_unused]] const Point<dim> &x,
                    [[maybe_unused]] const Solutions &sol) const
      {
      }

      //@}
      /**
       * @name Adaptivity indicators
       * Used by a mesh adaptor to decide which cells to refine. Each cell's indicator is the integral of
       * cell_indicator over the cell plus the integrals of face_indicator over its faces. Default: zero.
       */
      //@{

      /**
       * @brief Refinement indicator density on a face (DG, LDG).
       *
       * @param indicator to be filled: entry 0 is added to the cell of @p sol_s, entry 1 to the cell of @p sol_n.
       * @param normal the face normal, pointing out of the cell of @p sol_s.
       * @param p the point on the face.
       * @param sol_s, sol_n the solution on both sides, as named tuples. DG: `"fe_functions"`, `"fe_derivatives"`,
       * `"fe_hessians"`; LDG: `"fe_functions"`, `"LDG1"`, ... (values only).
       */
      template <int dim, typename NumberType, typename Solutions_s, typename Solutions_n>
      void face_indicator([[maybe_unused]] std::array<NumberType, 2> &indicator,
                          [[maybe_unused]] const Tensor<1, dim> &normal, [[maybe_unused]] const Point<dim> &p,
                          [[maybe_unused]] const Solutions_s &sol_s, [[maybe_unused]] const Solutions_n &sol_n) const
      {
      }

      /**
       * @brief Refinement indicator density in a cell (CG, DG, LDG).
       *
       * @param indicator the result.
       * @param p the point in the cell.
       * @param sol the solution there, as for face_indicator().
       */
      template <int dim, typename NumberType, typename Solution>
      void cell_indicator([[maybe_unused]] NumberType &indicator, [[maybe_unused]] const Point<dim> &p,
                          [[maybe_unused]] const Solution &sol) const
      {
      }

      //@}
      /**
       * @name Constraints
       */
      //@{

      /**
       * @brief Add affine constraints (dof = value) before sparsity patterns and operators are built.
       *
       * The default calls the model's `apply_boundary_affine_constraints(constraints, context)` and
       * `apply_affine_constraints(constraints, context)` if it has them; def::ConstrainOriginBoundaryPointToZero
       * and def::ConstrainOriginSupportPointToZero provide them. In such a function, `context.template
       * boundary<"u">()` lists the boundary dofs and their points of component "u", `context.template
       * support<"u">()` all its dofs.
       *
       * The origin helpers measure the distance from the origin by `x[0]` in 1D. In 2D the model must provide
       * `Model::OriginConstraintCoordinate<"u">::signed_coordinate(point)`, whose zero level set is constrained.
       */
      template <typename Constraints, typename Context>
      void affine_constraints(Constraints &constraints, const Context &context) const
      {
        if constexpr (requires(const Model &model, Constraints &constraint_matrix,
                               const Context &affine_constraint_context) {
                        model.apply_boundary_affine_constraints(constraint_matrix, affine_constraint_context);
                      })
          asImp().apply_boundary_affine_constraints(constraints, context);
        if constexpr (requires(const Model &model, Constraints &constraint_matrix,
                               const Context &affine_constraint_context) {
                        model.apply_affine_constraints(constraint_matrix, affine_constraint_context);
                      })
          asImp().apply_affine_constraints(constraints, context);
      }

      //@}
      /**
       * @name LDG levels
       * \f$L_k = \nabla\cdot F^{(k)}(L_{k-1}, x) + S^{(k)}(L_{k-1}, x)\f$ for \f$k \geq 1\f$, \f$L_0 = u\f$; see
       * model.hh.
       */
      //@{

      /**
       * @brief The flux \f$F^{(k)}\f$ of LDG level k = @p dependent. Default: zero.
       *
       * @tparam dependent the level k >= 1 being constructed.
       * @param F the result, one \f$d\f$-vector per component of level k.
       * @param x the point in field space.
       * @param u the values of level k - 1 at @p x.
       */
      template <uint dependent, int dim, typename NumberType, typename Vector, size_t n_fe_functions_dep>
      void ldg_flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions_dep> &F,
                    [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Vector &u) const
      {
      }

      /**
       * @brief The source \f$S^{(k)}\f$ of LDG level k = @p dependent. Default: zero.
       *
       * @tparam dependent the level k >= 1 being constructed.
       * @param s the result, one entry per component of level k.
       * @param x the point in field space.
       * @param u the values of level k - 1 at @p x.
       */
      template <uint dependent, int dim, typename NumberType, typename Vector, size_t n_fe_functions_dep>
      void ldg_source([[maybe_unused]] std::array<NumberType, n_fe_functions_dep> &s,
                      [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Vector &u) const
      {
      }

      /**
       * @brief Batched ldg_flux / ldg_source of level @p dependent, as evaluate_batch is for flux / source.
       *
       * The default calls ldg_flux and ldg_source point by point, as `out.requested(Term::...)` asks.
       * `batch.values(c)` is component c of level dependent - 1; write `out.flux(c, d)` and `out.source(c)` for
       * component c of level dependent.
       */
      template <uint dependent, typename Out, typename Batch>
      void ldg_evaluate_batch(Out &out, const Batch &batch) const
      {
        DiFfRG::internal::ldg_evaluate_per_point<dependent>(asImp(), out, batch);
      }

      //@}
    };

    namespace internal
    {
      template <typename> inline constexpr bool dependent_false_v = false;

      template <FixedString component_name, typename Model, int dim>
      double origin_constraint_coordinate(const Point<dim> &point)
      {
        if constexpr (dim == 1) {
          return point[0];
        } else {
          if constexpr (requires {
                          Model::template OriginConstraintCoordinate<component_name>::signed_coordinate(point);
                        }) {
            return Model::template OriginConstraintCoordinate<component_name>::signed_coordinate(point);
          } else {
            static_assert(dependent_false_v<Model>,
                          "Multidimensional origin affine-constraint helpers require Model::"
                          "OriginConstraintCoordinate<component_name>::signed_coordinate(point).");
            return 0.0;
          }
        }
      }

      template <FixedString component_name, typename Model, typename Context>
      std::vector<types::global_dof_index> select_origin_candidates([[maybe_unused]] const Context &context,
                                                                    const auto &view)
      {
        constexpr int dim = Context::dimension;
        static_assert(dim == 1 || dim == 2,
                      "Origin affine-constraint helpers currently support only one- and two-dimensional domains.");
        static_assert(Context::template component_size<component_name>() == 1,
                      "Origin affine-constraint helpers require a scalar FE-function component.");

        double best_abs_coordinate = std::numeric_limits<double>::infinity();
        bool has_non_negative_best = false;

        for (uint i = 0; i < view.dofs.n_elements(); ++i) {
          const double coordinate = origin_constraint_coordinate<component_name, Model>(view.points[i]);
          const double abs_coordinate = std::abs(coordinate);
          if (abs_coordinate < best_abs_coordinate) {
            best_abs_coordinate = abs_coordinate;
            has_non_negative_best = coordinate >= 0.0;
          } else if (abs_coordinate == best_abs_coordinate && coordinate >= 0.0) {
            has_non_negative_best = true;
          }
        }

        std::vector<types::global_dof_index> candidates;
        for (uint i = 0; i < view.dofs.n_elements(); ++i) {
          const double coordinate = origin_constraint_coordinate<component_name, Model>(view.points[i]);
          const double abs_coordinate = std::abs(coordinate);
          if (abs_coordinate != best_abs_coordinate) continue;
          if ((coordinate >= 0.0) != has_non_negative_best) continue;
          candidates.push_back(view.dofs.nth_index_in_set(i));
        }

        return candidates;
      }

      template <FixedString component_name, typename Context>
      std::optional<types::global_dof_index> select_origin_candidate(const Context &context, const auto &view)
      {
        constexpr int dim = Context::dimension;
        static_assert(dim == 1, "select_origin_candidate supports only one-dimensional domains; use "
                                "select_origin_candidates for multi-dimensional domains.");

        const auto candidates = select_origin_candidates<component_name, void>(context, view);
        if (candidates.empty()) return std::nullopt;
        return candidates.front();
      }
    } // namespace internal

    /**
     * @brief Constrain the boundary dofs of scalar FE function @p component_name closest to the origin to zero.
     *
     * The origin coordinate is `x[0]` in 1D, and `Model::OriginConstraintCoordinate<component_name>::
     * signed_coordinate(point)` in 2D. All boundary dofs with the smallest |coordinate| are constrained; if that is
     * attained on both sides, only those on the non-negative side.
     */
    template <FixedString component_name, typename Model> class ConstrainOriginBoundaryPointToZero
    {
    public:
      /// Called by AbstractModel::affine_constraints.
      template <typename Constraints, typename Context>
      void apply_boundary_affine_constraints(Constraints &constraints, const Context &context) const
      {
        const auto candidates = internal::select_origin_candidates<component_name, Model>(
            context, context.template boundary<component_name>());
        for (const auto dof : candidates) {
          constraints.add_line(dof);
          constraints.set_inhomogeneity(dof, 0.0);
        }
      }
    };

    /**
     * @brief As ConstrainOriginBoundaryPointToZero, but among all dofs of the component, not only boundary ones.
     *
     * For cell-centred layouts (DG0, FV), where the origin is not a boundary support point.
     */
    template <FixedString component_name, typename Model> class ConstrainOriginSupportPointToZero
    {
    public:
      /// Called by AbstractModel::affine_constraints.
      template <typename Constraints, typename Context>
      void apply_affine_constraints(Constraints &constraints, const Context &context) const
      {
        const auto candidates = internal::select_origin_candidates<component_name, Model>(
            context, context.template support<component_name>());
        for (const auto dof : candidates) {
          constraints.add_line(dof);
          constraints.set_inhomogeneity(dof, 0.0);
        }
      }
    };

    /// No affine constraints (same as inheriting nothing).
    template <typename Model> class NoAffineConstraints
    {
    };

    /**
     * @brief Keeps the time \f$t\f$ for models that are not RG flows. Use either this or def::fRG.
     */
    class Time
    {
    public:
      /// Set \f$t\f$; called by the timestepper.
      void set_time(double t);
      /// The current time \f$t\f$.
      const double &get_time() const;

    protected:
      // Initialized, like fRG::t below: a model is routinely built and read before the timestepper
      // first calls set_time(), and an indeterminate t is a bug that only shows up once the memory
      // happens to be dirty.
      double t = 0.;
    };

    /**
     * @brief Keeps the RG time \f$t\f$ and the scale \f$k = \Lambda e^{-t}\f$ with its powers, as protected members
     * `t`, `k`, `k2`, ..., `k6` and `Lambda`. Use either this or def::Time.
     */
    class fRG
    {
    public:
      /// @param Lambda the initial scale \f$\Lambda\f$.
      fRG(double Lambda);

      /// Reads \f$\Lambda\f$ from `/physical/Lambda`; if missing, uses 1 and warns.
      fRG(const ConfigTree &config);

      /// Set \f$t\f$ and update \f$k, k^2, \dots, k^6\f$; does nothing if \f$t\f$ is unchanged. Called by the
      /// timestepper.
      void set_time(double t);

      /// The current RG time \f$t\f$.
      const double &get_time() const;

    protected:
      const double Lambda;
      double t = 0., k = 0., k2 = 0., k3 = 0., k4 = 0., k5 = 0., k6 = 0.;
      bool time_initialized = false;
    };
  } // namespace def
} // namespace DiFfRG
