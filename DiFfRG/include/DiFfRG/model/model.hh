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

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief This namespace contains all default implementations and definitions needed for numerical models.
   *
   */
  namespace def
  {
    /// Block id for a component that carries no hyperbolic flux. @see AbstractModel::wave_speed_blocks.
    inline constexpr int no_wave_speed = -1;

    /**
     * @brief The abstract interface for any numerical model.
     * Most methods have a standard implementation, which can be overwritten if needed.
     * To see how the models are used, refer to the DiFfRG::AbstractAssembler class and the Numerical Models guide
     * in the documentation.
     *
     * @tparam Model The model which implements this interface. (CRTP)
     * @tparam Components_ The components of the model, this must be a DiFfRG::ComponentDescriptor.
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
      auto &components() { return m_components; }

    public:
      const auto &get_components() const { return m_components; }
      using Components = Components_;
      /**
       * @name Spatial discretization
       */
      //@{

      /**
       * @brief This method implements the initial condition for the FE functions.
       *
       * @note No standard implementation is given, this method has to be reimplemented whenever one uses FE functions.
       *
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param u_i the field values \f$u_i(x)\f$ at the point `x`.
       * This method should fill this argument with the desired initial condition.
       */
      template <int dim, typename Vector> void initial_condition(const Point<dim> &x, Vector &u_i) const = delete;

      /**
       * @brief
       * The mass function \f$m_i(\partial_t u_j, u_j, x)\f$ is implemented in this method.
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model. It is however necessary to leave at least the NumberType, Vector,
       * and Vector_dot template parameters, as these can differ between calls (e.g. when doing automatic
       * differentiation).
       *
       * @note The standard implementation of this method simply sets \f$m_i = \partial_t u_i\f$.
       *
       * @param m_i the resulting mass function \f$m_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param u_i the field values \f$u_i(x)\f$ at the point `x`.
       * @param dt_u_i the time derivative of the field values \f$\partial_t u_i(x)\f$ at the point `x`.
       */
      template <int dim, typename NumberType, typename Vector, typename Vector_dot, size_t n_fe_functions>
      void mass([[maybe_unused]] std::array<NumberType, n_fe_functions> &m_i, [[maybe_unused]] const Point<dim> &x,
                [[maybe_unused]] const Vector &u_i, const Vector_dot &dt_u_i) const
      {
        for (uint i = 0; i < n_fe_functions; ++i)
          m_i[i] = dt_u_i[i];
      }

      /**
       * @brief If not using a DAE, the mass matrix \f$m_{ij}(x)\f$ is implemented in this method.
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model. It is however necessary to leave at least the NumberType, Vector,
       * and Vector_dot template parameters, as these can differ between calls (e.g. when doing automatic
       * differentiation).
       *
       * @note The standard implementation of this method simply sets \f$m_{ij} = \delta_{ij}\f$.
       *
       * @param m_ij the resulting mass matrix \f$m_{ij}\f$, with \f$N_f\f$ components in each dimension.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       */
      template <int dim, typename NumberType, size_t n_fe_functions>
      void mass(std::array<std::array<NumberType, n_fe_functions>, n_fe_functions> &m_ij,
                [[maybe_unused]] const Point<dim> &x) const
      {
        for (uint i = 0; i < n_fe_functions; ++i)
          for (uint j = 0; j < n_fe_functions; ++j)
            m_ij[i][j] = 0.;
        for (uint i = 0; i < n_fe_functions; ++i)
          m_ij[i][i] = 1.;
      }

      /**
       * @brief The flux function \f$F_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x)\f$ is implemented by this
       * method.
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model. It is however necessary to leave at least the NumberType, Vector,
       * and Vector_dot template parameters, as these can differ between calls (e.g. when doing automatic
       * differentiation).
       *
       * @note The standard implementation of this method simply sets \f$F_i = 0\f$.
       *
       * @note The meaning of this method depends on the discretization. For CG/DG/LDG it is the full flux of the
       * conservation law. For the Kurganov Tadmor scheme it is the *advection* flux, i.e. the hyperbolic part which
       * determines the numerical face flux and the wave speeds; the parabolic part is implemented separately in
       * `diffusion_flux`. In the Kurganov Tadmor case the assembler evaluates this callback separately for the minus
       * and plus face traces, and the derivatives are the face gradients produced by the active advection
       * reconstructor; they are distinct from the corrected gradients handed to `diffusion_flux`.
       *
       * @param F_i the resulting flux function \f$F_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param sol a named tuple; which entries it carries depends on the assembler, see source().
       *
       * @note The extractors are those of the last extract() call, i.e. of the state the assembler is
       * currently linearising about, and they are plain numbers rather than AD types. Their dependence
       * on the FE solution is therefore absent from the flux jacobian -- extractors are frozen within a
       * Newton step, exactly as they are for source(). A model that solves for something at the EoM in
       * extract() and reads it back here gets an approximate jacobian, which costs iterations rather
       * than accuracy: the extraction is redone for every residual.
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief If the Kurganov Tadmor Scheme is used, this is the implementation of the diffusion (parabolic) part of
       * the face flux. \f$D_i(u_j, \partial_x u_j, \partial_x^3 u_j, x)\f$
       *
       * @remarks The assembler evaluates this callback separately for the minus and plus face traces, using the
       * corrected gradients of the diffusion reconstructor; these are distinct from the advection face gradients
       * handed to `flux`. This method is never called by the CG/DG/LDG discretizations.
       *
       * @note The standard implementation of this method simply sets \f$D_i = 0\f$.
       *
       * @note Sign convention: the face flux is \f$(H + D)\cdot n\f$, i.e. the advection numerical flux \f$H\f$ (built
       * from `flux`) and the diffusion flux \f$D\f$ (from this method) are SUMMED. Both methods therefore return the
       * physical flux with the same sign - exactly the conservation-law convention used by CG / LLFFlux. A diffusion
       * flux \f$f_{diff}\f$ must be a DECREASING function of the gradient (\f$\partial f_{diff} / \partial (\partial
       * u) < 0\f$) for forward diffusion, e.g. \f$f_{diff} = -\nu\, \partial u\f$ for the heat/viscous term.
       *
       * @param F_i the resulting diffusion flux \f$D_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param sol a `std::tuple<...>` which contains
       * 1. the array u_j
       * 2. the array of arrays \f$\partial_x u_j\f$
       * 3. the array of arrays of arrays \f$\partial_x^3 u_j\f$
       * 4. the array of extractors \f$e_b\f$
       * 5. the array of variables \f$v_a\f$
       *
       * @note Extractors are frozen w.r.t. the FE solution here too. @see flux.
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void diffusion_flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                          [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief The source function \f$s_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x)\f$ is implemented by
       * this method.
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the
       * types are consistent with the rest of the model. It is however necessary to leave at least the NumberType,
       * Vector, and Vector_dot template parameters, as these can differ between calls (e.g. when doing automatic
       * differentiation).
       *
       * @note The standard implementation of this method simply sets \f$s_i = 0\f$.
       *
       * @param s_i the resulting source function \f$s_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param sol a named tuple; which entries it carries depends on the assembler:
       * - CG and DG: `"fe_functions"` \f$u_j\f$, `"fe_derivatives"` \f$\partial_x u_j\f$,
       *   `"fe_hessians"` \f$\partial_x^2 u_j\f$, `"extractors"` \f$e_b\f$, `"variables"` \f$v_a\f$; no
       *   hessians (derivatives) if the model declares `batch_reads_hessians` (`batch_reads_derivatives`) false
       * - LDG main level: `"fe_functions"`, `"LDG1"`, ..., `"extractors"`, `"variables"`
       * - KT-FV: `"fe_functions"`, `"fe_derivatives"`, `"extractors"`, `"variables"` -- no hessians, and
       *   the derivatives are the scheme's reconstructed cell gradient
       *
       * Access the entries by name (`get<"fe_derivatives">(sol)`); a model that reads an entry its
       * assembler does not provide fails to compile.
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void source([[maybe_unused]] std::array<NumberType, n_fe_functions> &s_i, [[maybe_unused]] const Point<dim> &x,
                  [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief The flux, source and/or diffusion flux at all points of a batch at once. Every assembler evaluates
       * the model through this (CG, DG, the main level of LDG, and the face traces and cells of KT); see
       * Tutorial 6.
       *
       * The assembler requests only the terms it uses (`out.requested(Term::flux)` etc.): fluxes at faces, flux
       * and source at cell points; KT asks for one term per call (flux, diffusion_flux or source). The standard
       * implementation evaluates the requested per-point callbacks (flux(), source(), diffusion_flux()) in one
       * flat parallel loop over the batch. Override it to evaluate the expensive part, typically the momentum
       * integrals, for all points in one go with an integrator's map_points(), which also runs on the GPU; skip
       * the terms that were not requested. A model deriving from def::AD then gets its jacobians from the same
       * function, and def::LLFFlux / def::FlowBoundaries evaluate the face fluxes through it.
       *
       * The number type of the batch (`typename Batch::number_type`) is double or an AD type, and the batch may
       * hold several stacked copies of the assembler's points (AD seeds, LLF perturbations): size everything
       * from `batch.size()` and never keep per-point state across calls.
       *
       * @param out the result, a BatchOutput: `out.flux(c, d)`, `out.source(c)` and `out.diffusion_flux(c, d)`
       * are the columns (PointSpan) of component c (and direction d) over all points, zero on entry; asking for a
       * term that was not requested throws.
       * @param batch the solution at the points: columns `values(c)`, `derivatives(c, d)`, `hessians(c, d1, d2)`,
       * `coordinates(d)` (each a PointSpan, directly usable as a per-point argument of map_points()), plus
       * `extractors()`, `variables()`, `size()`, `x(i)` and `cell_width(i)`. Which inputs exist depends on the
       * assembler and is known at compile time (`Batch::has_derivatives`, `Batch::has_hessians`; reading an
       * absent one does not compile): LDG has values and `ldg_values(k, c)` for component c
       * of level k, KT's diffusion traces add `third_derivatives(c, d0, d1, d2)` (`Batch::with_third`).
       *
       * @note Set `static constexpr bool batch_reads_hessians = false` (or `batch_reads_derivatives`) in the model
       * when no term reads them: CG and DG then skip gathering them, and the AD jacobians skip their seeds.
       */
      template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
      {
        DiFfRG::internal::evaluate_per_point(asImp(), out, batch);
      }

      /**
       * @brief A method to find out which components of the mass function are differential when using a DAE.
       *
       * @note The standard implementation of this method tests whether the mass function changes when changing the time
       * derivative of one component slightly. For highly complicated models, this method might not be able to set all
       * differential components correctly.
       *
       * @return std::vector<bool> with `true` for differential components and `false` for algebraic components.
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
       * @brief Which FE components share a wave speed, and which carry no hyperbolic flux at all.
       *
       * Only the Kurganov-Tadmor assembler consults this. KT adds a dissipation term
       * -a/2 (u^+ - u^-) to the numerical flux of every component, with `a` a wave speed read off
       * the flux jacobian. Which jacobian, and whether a component is dissipated at all, is what
       * this declares: components sharing a block id share one speed, computed as the spectral
       * radius of the flux jacobian *restricted to that block*, and a component marked
       * `no_wave_speed` is not dissipated.
       *
       * Two things go wrong when the whole system shares one speed, and a model that mixes a
       * conservation law with constraints needs both fixed.
       *
       * A component whose flux is identically zero is dissipated all the same. For a differential
       * component that is harmless: its diagonal carries c_j * JxW, which swamps the dissipation. An
       * algebraic component has no such term, so the dissipation becomes the leading entry of its
       * row -- and drags the reconstruction's slope limiter into the newton jacobian with it, where
       * a limiter kink stalls the iteration outright. `no_wave_speed` drops the dissipation instead,
       * leaving the row exactly as the model wrote it.
       *
       * A component that *does* carry a flux, but a much slower one than the fastest in the system,
       * is over-dissipated by the ratio of the two speeds. That is worse than inaccurate when the
       * row is algebraic. Its own terms are integrated over the cell and so carry a factor of the
       * cell width, while the dissipation is a face quantity and does not; the row's diagonal
       * dominance is then O(dx) and the row *degenerates under refinement*, so the scheme fails when
       * the grid is made finer. Giving such a component its own block restores the balance: both the
       * diagonal and the off-diagonal scale as a/dx and their difference stays finite.
       *
       * Blocking is a statement that the cross-couplings between blocks do not carry characteristics
       * -- true when the off-block rows are constraints rather than conservation laws, which is the
       * case this exists for. It is not a licence to split a genuinely hyperbolic system, where the
       * full spectral radius is the safe choice and is what one block gives.
       *
       * This is a structural property of the model, not a function of the state: it is queried per
       * face, and blocks that varied with the solution would be both a cost and a discontinuity.
       *
       * The default puts every component in one block, i.e. the scheme as it was before this hook
       * existed.
       */
      template <size_t n_fe_functions> void wave_speed_blocks(std::array<int, n_fe_functions> &blocks) const
      {
        blocks.fill(0);
      }

      //@}
      /**
       * @name Other variables
       */
      //@{

      template <typename Vector> void initial_condition_variables([[maybe_unused]] Vector &v_a) const
      {
        // Just to avoid warnings
      }

      template <typename Vector, typename Solution>
      void dt_variables([[maybe_unused]] Vector &r_a, [[maybe_unused]] const Solution &sol) const
      {
        // Just to avoid warnings
      }

      //@}
      /**
       * @name Extractors
       */
      //@{

      /**
       * @brief Read data off the FE solution at a single point and hand it to the Variables.
       *
       * This is the only bridge from the field-space (FE) sector into the Variables sector: the
       * values stored in @p result are what `dt_variables` sees under `get<"extractors">(sol)`.
       *
       * @param x The point the extractors are evaluated at. By default this is the EoM; a model can
       *          choose otherwise by defining `extractor_point` (see DiFfRG::HasExtractorPoint).
       */
      template <int dim, typename Vector, typename Solutions>
      void extract([[maybe_unused]] Vector &result, [[maybe_unused]] const Point<dim> &x,
                   [[maybe_unused]] const Solutions &sol) const
      {
      }

      //@}
      /**
       * @name LDG equations
       */
      //@{

      /**
       * @brief The LDG flux function \f$F^{LDG}_i(u_j, x),\,i>0\f$ is implemented by this method.
       *
       * The assembler constructs the i-th LDG function l_i from the i-1-th level
       * as \f[l_i = \partial_x F^{LDG}_i(l_{i-1}, x) + s^{LDG}_i(l_{i-1}, x)\f]
       * Here, \f$l_0\f$ is the solution itself (with all its components).
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model.
       *
       * @note The standard implementation of this method simply sets \f$F^{LDG}_i = 0\f$.
       *
       * @tparam dependent the index \f$i\f$ of the dependent variable \f$l_i\f$ which is constructed from the previous
       * level \f$l_{i-1}\f$.
       *
       * @param F the resulting LDG flux function \f$F^{LDG}_i\f$, with n_fe_functions_dep components.
       * This method should fill this argument with the desired structure of the flow equation.
       *
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param u the field values of \f$l_j(x)\f$ at the point `x`.
       *
       */
      template <uint dependent, int dim, typename NumberType, typename Vector, size_t n_fe_functions_dep>
      void ldg_flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions_dep> &F,
                    [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Vector &u) const
      {
      }

      /**
       * @brief The LDG source function \f$s^{LDG}_i(u_j, x),\,i>0\f$ is implemented by this method.
       *
       * The assembler constructs the i-th LDG function l_i from the i-1-th level
       * as \f[l_i = \partial_x F^{LDG}_i(l_{i-1}, x) + s^{LDG}_i(l_{i-1}, x)\f]
       * Here, \f$l_0\f$ is the solution itself (with all its components).
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model.
       *
       * @note The standard implementation of this method simply sets \f$s^{LDG}_i = 0\f$.
       *
       * @tparam dependent the index \f$i\f$ of the dependent variable \f$l_i\f$ which is constructed from the previous
       * level \f$l_{i-1}\f$.
       *
       * @param s the resulting LDG source function \f$s^{LDG}_i\f$, with n_fe_functions_dep components.
       * This method should fill this argument with the desired structure of the flow equation.
       *
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param u the field values of \f$l_j(x)\f$ at the point `x`.
       *
       */
      template <uint dependent, int dim, typename NumberType, typename Vector, size_t n_fe_functions_dep>
      void ldg_source([[maybe_unused]] std::array<NumberType, n_fe_functions_dep> &s,
                      [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Vector &u) const
      {
      }

      /**
       * @brief LDG level `dependent` at all quadrature points of a batch of level dependent - 1 at once: the
       * batched counterpart of ldg_flux and ldg_source, which the LDG assembler calls.
       *
       * The standard implementation evaluates the requested per-point ldg_flux / ldg_source (see
       * `out.requested(Term::...)`) in one flat parallel loop. Override it to evaluate them for all points in one
       * go. `batch.values(c)` is component c of level dependent - 1, a PointSpan over all points; `out.flux(c, d)`
       * and `out.source(c)` are the columns of component c of level dependent.
       */
      template <uint dependent, typename Out, typename Batch>
      void ldg_evaluate_batch(Out &out, const Batch &batch) const
      {
        DiFfRG::internal::ldg_evaluate_per_point<dependent>(asImp(), out, batch);
      }

      template <int dim, typename NumberType, typename Solutions_s, typename Solutions_n>
      void face_indicator([[maybe_unused]] std::array<NumberType, 2> &indicator,
                          [[maybe_unused]] const Tensor<1, dim> &normal, [[maybe_unused]] const Point<dim> &p,
                          [[maybe_unused]] const Solutions_s &sol_s, [[maybe_unused]] const Solutions_n &sol_n) const
      {
      }

      template <int dim, typename NumberType, typename Solution>
      void cell_indicator([[maybe_unused]] NumberType &indicator, [[maybe_unused]] const Point<dim> &p,
                          [[maybe_unused]] const Solution &sol) const
      {
      }

      template <int dim, typename Vector>
      std::array<double, dim> EoM([[maybe_unused]] const Point<dim> &x, const Vector &u) const
      {
        return std::array<double, dim>{{u[0]}};
      }

      /**
       * @brief Whether extract() reads the reconstructed potential handed to it.
       *
       * The potential slots of the tuple passed to extract() are filled by reconstructing a scalar potential
       * from raw_potential_gradient() over the whole mesh -- a direct solve, run on every residual and
       * jacobian evaluation, and wasted on the many models whose extractors need only the solution. A model
       * that never reads those slots should set this to false; the slots are then filled with a type that has
       * no operations, so reading one is a compile error rather than a silent zero. Readouts are unaffected:
       * the potential is written to the output there regardless.
       */
      static constexpr bool extract_uses_potential = true;

      /**
       * @brief The unmodified gradient of the scalar potential reconstructed for readouts and extractors.
       *
       * This is deliberately separate from the EoM callback supplied by readouts_multiple(): a physical EoM may
       * contain explicit-breaking or other terms which are not part of the raw potential. By default, the first dim
       * solution components are interpreted as the raw potential gradient; missing components are zero-filled. Models
       * with a different component layout should override this method.
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
       * @brief Relocate the point found by the EoM search, given the solution values there.
       *
       * Pointwise, and therefore limited to decisions that can be made from the located point alone
       * -- e.g. freezing the EoM once it jumps backwards. A model that has to inspect the solution
       * profile as a whole to decide where its extractors should be read instead defines
       * `extractor_point`; see DiFfRG::HasExtractorPoint.
       */
      template <int dim, typename Vector> Point<dim> EoM_postprocess(const Point<dim> &EoM, const Vector &) const
      {
        return EoM;
      }

      template <typename FUN, typename DataOut> void readouts_multiple(FUN &helper, DataOut &) const
      {
        helper(
            "primary", [&](const auto &x, const auto &u_i) { return asImp().EoM(x, u_i); }, // chiral EoM
            [&](auto &output, const auto &x, const auto &sol) { asImp().readouts(output, x, sol); });
      }

      template <int dim, typename DataOut, typename Solutions>
      void readouts([[maybe_unused]] DataOut &output, [[maybe_unused]] const Point<dim> &x,
                    [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief Add affine constraints to the FE/DG system before sparsity patterns and operators are rebuilt.
       *
       * Boundary-only constraints should use `apply_boundary_affine_constraints(constraints, context)` and inspect
       * `context.template boundary<"u">()`. Constraints that may need interior support points should use
       * `apply_affine_constraints(constraints, context)` and inspect `context.template support<"u">()`.
       *
       * The origin helpers use `x[0]` as their signed origin coordinate in one-dimensional domains. In
       * multidimensional domains, models using these helpers must provide
       * `Model::OriginConstraintCoordinate<component_name>::signed_coordinate(point)` to define the zero level set
       * that should be constrained for each named component.
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
     * @brief Constrain the boundary dofs of a named scalar FE-function component nearest its origin coordinate to zero.
     *
     * In one dimension, the origin coordinate is `x[0]`. In multidimensional domains, `Model` must provide an
     * `OriginConstraintCoordinate<component_name>` policy whose `signed_coordinate(point)` method defines the zero
     * level set. All boundary dofs on the nearest discrete zero level set are constrained, with symmetric ties
     * resolved toward the non-negative side.
     */
    template <FixedString component_name, typename Model> class ConstrainOriginBoundaryPointToZero
    {
    public:
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
     * @brief Constrain the support dofs of a named scalar FE-function component nearest its origin coordinate to zero.
     *
     * This is useful for cell-centered DG0/FV layouts where `sigma = 0` is not itself a boundary support point.
     * In multidimensional domains, `Model::OriginConstraintCoordinate<component_name>::signed_coordinate(point)`
     * defines the zero level set to constrain.
     */
    template <FixedString component_name, typename Model> class ConstrainOriginSupportPointToZero
    {
    public:
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

    template <typename Model> class NoAffineConstraints
    {
    };

    class Time
    {
    public:
      void set_time(double t);
      const double &get_time() const;

    protected:
      // Initialized, like fRG::t below: a model is routinely built and read before the timestepper
      // first calls set_time(), and an indeterminate t is a bug that only shows up once the memory
      // happens to be dirty.
      double t = 0.;
    };

    /**
     * @brief The fRG class is used to keep track of the RG time and the cutoff scale.
     */
    class fRG
    {
    public:
      /**
       * @brief Construct a new fRG object from a given initial cutoff scale
       *
       * @param Lambda the initial cutoff scale of the fRG
       */
      fRG(double Lambda);

      /**
       * @brief Construct a new fRG object from a given ConfigTree object
       *
       * @param config the ConfigTree object containing the initial cutoff scale at "/physical/Lambda"
       */
      fRG(const ConfigTree &config);

      /**
       * @brief Set the time of the fRG object, updating the cutoff scale and its powers
       *
       * @param t the time to set
       */
      void set_time(double t);

      /**
       * @brief Get the time of the fRG object
       *
       * @return const double& the time of the fRG object
       */
      const double &get_time() const;

    protected:
      const double Lambda;
      double t = 0., k = 0., k2 = 0., k3 = 0., k4 = 0., k5 = 0., k6 = 0.;
      bool time_initialized = false;
    };
  } // namespace def
} // namespace DiFfRG
