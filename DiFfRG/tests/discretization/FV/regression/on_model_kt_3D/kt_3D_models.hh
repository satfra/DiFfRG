#pragma once

// The 7 LSM physical-scale models exercised by the 3D regression suite.
// Each model uses the same physics convention:
//   V₀(σ) = (m²/2) σ² + (λ/16) σ⁴, equivalently V₀(ρ) = m²·ρ + (λ/4)·ρ².
//   ρ-coords initial: u(ρ) = ∂V/∂ρ = m² + (λ/2)·ρ.
//   σ-coords initial: u(σ) = ∂V/∂σ = m²·σ + (λ/4)·σ³.
// Both reach the same physical u at the matching point σ_max = sqrt(2·ρ_max).
//
// Models (see plan in /home/franz/.claude/plans/graceful-mapping-mist.md):
//   LSM_sigma_analytic           σ   analytic Litim T=0  closed form k⁵/(12π²·sqrt(k²+m²))
//   LSM_sigma_integrator_PolyExp σ   integrator PolyExp T=0.05 (matches Examples/ONfiniteT)
//   LSM_sigma_integrator_Litim   σ   integrator Litim T=0
//   LSM_rho_analytic             ρ   analytic Litim T=0
//   LSM_rho_integrator_PolyExp   ρ   integrator PolyExp T=0.05 (matches Examples/ONfiniteT)
//   LSM_rho_integrator_Litim     ρ   integrator Litim T=0
//   LSM_CG                       ρ   integrator PolyExp T=0.05 + CG (matches Examples/ONfiniteT)

// IMPORTANT: setup.hh (which pulls in KurganovTadmor.hh, with its
// `Quadrature<dim>` use that resolves to dealii::Quadrature) MUST come
// before kernels.hh (which pulls in DiFfRG/physics/integration.hh, which
// introduces DiFfRG::Quadrature<NT> and would otherwise shadow the dealii
// one during KT-assembler template parsing).
#include "kt_3D_setup.hh"
#include "kt_3D_kernels.hh"

namespace on_kt_3D
{
  using std::get;

  // --- σ-coord analytic (Litim T=0 closed form) ------------------------------

  class LSM_sigma_analytic : public def::AbstractModel<LSM_sigma_analytic, Components>,
                             public def::fRG,
                             public def::OriginOddLinearExtrapolationBoundaries<LSM_sigma_analytic>,
                             public ConstrainUAtOrigin<LSM_sigma_analytic>,
                             public def::AD<LSM_sigma_analytic>
  {
  public:
    LSM_sigma_analytic(const ConfigTree &json, const GridSettings & /*grid*/) : def::fRG(json) {}

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      // u(σ) = ∂V/∂σ = m²·σ + (λ/4)·σ³  (matches V(σ) = m²/2·σ² + λ/16·σ⁴).
      const double sigma = x[0];
      values[0] = LSMPhysicalParameters::m2 * sigma + 0.25 * LSMPhysicalParameters::lambda_quartic * sigma * sigma * sigma;
    }

    // F_pion = (N-1) · k⁵ / (12π² · sqrt(k² + m²_pion))  with m²_pion = u/σ.
    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "LSM_sigma_analytic is one-dimensional.");
      static_assert(n_fe_functions == 1);
      const auto &fe_functions = get<0>(sol);
      const NT u = fe_functions[0];
      const double sigma = x[0];
      const double k2 = k * k;
      const double prefactor = (k * k * k * k * k) / (12.0 * M_PI * M_PI);
      const NT m2Pi = (std::abs(sigma) < origin_tol) ? NT(0.0) : (u / NT(sigma));
      F_i[0][0] = NT(prefactor * (LSMPhysicalParameters::N - 1.0)) / sqrt(NT(k2) + m2Pi);
    }

    // F_sigma = k⁵ / (12π² · sqrt(k² + m²_sigma))  with m²_sigma = ∂u/∂σ.
    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void diffusion_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> & /*x*/,
                        const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "LSM_sigma_analytic is one-dimensional.");
      static_assert(n_fe_functions == 1);
      const auto &fe_derivatives = get<1>(sol);
      const NT du_dsigma = fe_derivatives[0][0];
      const double k2 = k * k;
      const double prefactor = (k * k * k * k * k) / (12.0 * M_PI * M_PI);
      // KT residual sums fluxes: (H + D)·n. Diffusion flux = physical flux, same sign as
      // advection (CG convention). ∝ +1/√ is decreasing in the gradient ⇒ forward.
      F_i[0][0] = NT(prefactor) / sqrt(NT(k2) + du_dsigma);
    }
  };

  // --- ρ-coord analytic (Litim T=0 closed form, identical to σ but in ρ-vars) ---

  class LSM_rho_analytic : public def::AbstractModel<LSM_rho_analytic, Components>,
                           public def::fRG,
                           public def::RhoSymmetricLinearExtrapolationBoundaries<LSM_rho_analytic>,
                           public def::AD<LSM_rho_analytic>
  {
  public:
    LSM_rho_analytic(const ConfigTree &json, const GridSettings & /*grid*/) : def::fRG(json) {}

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      // u(ρ) = ∂V/∂ρ = m² + (λ/2)·ρ.
      const double rho = x[0];
      values[0] = LSMPhysicalParameters::m2 + 0.5 * LSMPhysicalParameters::lambda_quartic * rho;
    }

    // F_pion = (N-1) · k⁵ / (12π² · sqrt(k² + m²_pion))  with m²_pion = u (no 1/σ in ρ-form).
    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> & /*x*/,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "LSM_rho_analytic is one-dimensional.");
      static_assert(n_fe_functions == 1);
      const auto &fe_functions = get<0>(sol);
      const NT m2Pi = fe_functions[0];
      const double k2 = k * k;
      const double prefactor = (k * k * k * k * k) / (12.0 * M_PI * M_PI);
      F_i[0][0] = NT(prefactor * (LSMPhysicalParameters::N - 1.0)) / sqrt(NT(k2) + m2Pi);
    }

    // F_sigma = k⁵ / (12π² · sqrt(k² + m²_sigma))  with m²_sigma = m² + 2ρ·∂m²/∂ρ.
    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void diffusion_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
                        const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "LSM_rho_analytic is one-dimensional.");
      static_assert(n_fe_functions == 1);
      const auto &fe_functions = get<0>(sol);
      const auto &fe_derivatives = get<1>(sol);
      const double rho = x[0];
      const NT m2Sigma = fe_functions[0] + NT(2.0 * rho) * fe_derivatives[0][0];
      const double k2 = k * k;
      const double prefactor = (k * k * k * k * k) / (12.0 * M_PI * M_PI);
      // Same sign as advection: KT sums (H + D)·n (see σ-analytic note above).
      F_i[0][0] = NT(prefactor) / sqrt(NT(k2) + m2Sigma);
    }
  };

  // --- σ-coord integrator, PolyExp + finite T (matches Examples/ONfiniteT/model_KT_sigma.hh) ---

  template <typename FlowsType>
  class LSM_sigma_integrator_base : public def::AbstractModel<LSM_sigma_integrator_base<FlowsType>, Components>,
                                    public def::fRG,
                                    public def::OriginOddLinearExtrapolationBoundaries<LSM_sigma_integrator_base<FlowsType>>,
                                    public ConstrainUAtOrigin<LSM_sigma_integrator_base<FlowsType>>,
                                    public def::AD<LSM_sigma_integrator_base<FlowsType>>
  {
  public:
    LSM_sigma_integrator_base(const ConfigTree &json, const GridSettings & /*grid*/)
        : def::fRG(json), flows_(json), T_(integrator_T())
    {
      flows_.set_k(this->Lambda);
      flows_.set_T(T_);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double sigma = x[0];
      values[0] = LSMPhysicalParameters::m2 * sigma + 0.25 * LSMPhysicalParameters::lambda_quartic * sigma * sigma * sigma;
    }

    void set_time(double t_)
    {
      this->t = t_;
      this->k = std::exp(-t_) * this->Lambda;
      flows_.set_k(this->k);
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim);
      static_assert(n_fe_functions == 1);
      const auto &fe_functions = get<0>(sol);
      const NT u = fe_functions[0];
      const double sigma = x[0];
      const NT m2Pi = (std::abs(sigma) < origin_tol) ? NT(0.0) : (u / NT(sigma));
      NT pion_loop;
      flows_.V_pion.get(pion_loop, this->k, LSMPhysicalParameters::N, T_, m2Pi);
      F_i[0][0] = (LSMPhysicalParameters::N - 1.0) * pion_loop;
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void diffusion_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> & /*x*/,
                        const Solutions &sol) const
    {
      static_assert(spatial_dim == dim);
      static_assert(n_fe_functions == 1);
      const auto &fe_derivatives = get<1>(sol);
      const NT m2Sigma = fe_derivatives[0][0];
      NT sigma_loop;
      flows_.V_sigma.get(sigma_loop, this->k, LSMPhysicalParameters::N, T_, m2Sigma);
      // Same sign as advection: KT sums (H + D)·n; see note in quark_meson_kt_regression.cc.
      F_i[0][0] = sigma_loop;
    }

  private:
    // PolyExp variant runs at the physical T from LSMPhysicalParameters; Litim
    // variant runs at T=0 (its closed-form match). Specialised below.
    static constexpr double integrator_T();

    mutable FlowsType flows_;
    double T_;
  };

  template <> constexpr double LSM_sigma_integrator_base<PolyExpFlows>::integrator_T() { return LSMPhysicalParameters::T; }
  template <> constexpr double LSM_sigma_integrator_base<LitimT0Flows>::integrator_T() { return 0.0; }

  using LSM_sigma_integrator_PolyExp = LSM_sigma_integrator_base<PolyExpFlows>;
  using LSM_sigma_integrator_Litim = LSM_sigma_integrator_base<LitimT0Flows>;

  // --- ρ-coord integrator (matches Examples/ONfiniteT/model_KT.hh) -----------

  template <typename FlowsType>
  class LSM_rho_integrator_base : public def::AbstractModel<LSM_rho_integrator_base<FlowsType>, Components>,
                                  public def::fRG,
                                  public def::RhoSymmetricLinearExtrapolationBoundaries<LSM_rho_integrator_base<FlowsType>>,
                                  public def::AD<LSM_rho_integrator_base<FlowsType>>
  {
  public:
    LSM_rho_integrator_base(const ConfigTree &json, const GridSettings & /*grid*/)
        : def::fRG(json), flows_(json), T_(integrator_T())
    {
      flows_.set_k(this->Lambda);
      flows_.set_T(T_);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double rho = x[0];
      values[0] = LSMPhysicalParameters::m2 + 0.5 * LSMPhysicalParameters::lambda_quartic * rho;
    }

    void set_time(double t_)
    {
      this->t = t_;
      this->k = std::exp(-t_) * this->Lambda;
      flows_.set_k(this->k);
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> & /*x*/,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim);
      static_assert(n_fe_functions == 1);
      const auto &fe_functions = get<0>(sol);
      const NT m2Pi = fe_functions[0];
      NT pion_loop;
      flows_.V_pion.get(pion_loop, this->k, LSMPhysicalParameters::N, T_, m2Pi);
      F_i[0][0] = (LSMPhysicalParameters::N - 1.0) * pion_loop;
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void diffusion_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
                        const Solutions &sol) const
    {
      static_assert(spatial_dim == dim);
      static_assert(n_fe_functions == 1);
      const auto &fe_functions = get<0>(sol);
      const auto &fe_derivatives = get<1>(sol);
      const double rho = x[0];
      const NT m2Sigma = fe_functions[0] + NT(2.0 * rho) * fe_derivatives[0][0];
      NT sigma_loop;
      flows_.V_sigma.get(sigma_loop, this->k, LSMPhysicalParameters::N, T_, m2Sigma);
      // Same sign as advection: KT sums (H + D)·n; see note in quark_meson_kt_regression.cc.
      F_i[0][0] = sigma_loop;
    }

  private:
    static constexpr double integrator_T();

    mutable FlowsType flows_;
    double T_;
  };

  template <> constexpr double LSM_rho_integrator_base<PolyExpFlows>::integrator_T() { return LSMPhysicalParameters::T; }
  template <> constexpr double LSM_rho_integrator_base<LitimT0Flows>::integrator_T() { return 0.0; }

  using LSM_rho_integrator_PolyExp = LSM_rho_integrator_base<PolyExpFlows>;
  using LSM_rho_integrator_Litim = LSM_rho_integrator_base<LitimT0Flows>;

  // --- ρ-coord large-N "all-pion" KT model -----------------------------------
  //
  // The sigma (radial) mode is replaced by another pion: all N modes are
  // treated as Goldstones with mass m²_pion = u (no derivative dependence).
  //   F_total = N · V_pion(u)
  // This eliminates the derivative-dependent diffusion flux entirely — the
  // PDE becomes purely hyperbolic (first-order advection), which is the
  // setting KT was designed for. Diagnostic: if KT reaches t=4 here but
  // fails on the full O(N) split, the derivative-dependent diffusion flux
  // (or its interaction with KT's face reconstruction) is the discriminator.
  //
  // KT split: all flux goes in the advection slot (depends on u only); the
  // diffusion slot is identically zero.
  template <typename FlowsType>
  class LSM_rho_largeN_base : public def::AbstractModel<LSM_rho_largeN_base<FlowsType>, Components>,
                              public def::fRG,
                              public def::RhoSymmetricLinearExtrapolationBoundaries<LSM_rho_largeN_base<FlowsType>>,
                              public def::AD<LSM_rho_largeN_base<FlowsType>>
  {
  public:
    LSM_rho_largeN_base(const ConfigTree &json, const GridSettings & /*grid*/)
        : def::fRG(json), flows_(json), T_(integrator_T())
    {
      flows_.set_k(this->Lambda);
      flows_.set_T(T_);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double rho = x[0];
      values[0] = LSMPhysicalParameters::m2 + 0.5 * LSMPhysicalParameters::lambda_quartic * rho;
    }

    void set_time(double t_)
    {
      this->t = t_;
      this->k = std::exp(-t_) * this->Lambda;
      flows_.set_k(this->k);
    }

    // F_adv = N · V_pion(u)  (all N modes have m²_pion = u).
    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> & /*x*/,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim);
      static_assert(n_fe_functions == 1);
      const auto &fe_functions = get<0>(sol);
      const NT m2Pi = fe_functions[0];
      NT pion_loop;
      flows_.V_pion.get(pion_loop, this->k, LSMPhysicalParameters::N, T_, m2Pi);
      F_i[0][0] = LSMPhysicalParameters::N * pion_loop;
    }

    // No derivative-dependent (diffusion) flux: purely hyperbolic PDE.
    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void diffusion_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> & /*x*/,
                        const Solutions & /*sol*/) const
    {
      static_assert(spatial_dim == dim);
      static_assert(n_fe_functions == 1);
      F_i[0][0] = NT(0.0);
    }

  private:
    static constexpr double integrator_T();

    mutable FlowsType flows_;
    double T_;
  };

  template <> constexpr double LSM_rho_largeN_base<PolyExpFlows>::integrator_T() { return LSMPhysicalParameters::T; }
  template <> constexpr double LSM_rho_largeN_base<LitimT0Flows>::integrator_T() { return 0.0; }

  using LSM_rho_largeN_PolyExp = LSM_rho_largeN_base<PolyExpFlows>;
  using LSM_rho_largeN_Litim = LSM_rho_largeN_base<LitimT0Flows>;

  // CG counterpart of the large-N model (combined flux F = N · V_pion(u)).
  class LSM_CG_largeN : public def::AbstractModel<LSM_CG_largeN, Components>,
                        public def::fRG,
                        public def::LLFFlux<LSM_CG_largeN>,
                        public def::FlowBoundaries<LSM_CG_largeN>,
                        public def::AD<LSM_CG_largeN>
  {
  public:
    LSM_CG_largeN(const ConfigTree &json, const GridSettings & /*grid*/) : def::fRG(json), flows_(json)
    {
      flows_.set_k(this->Lambda);
      flows_.set_T(LSMPhysicalParameters::T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double rho = x[0];
      values[0] = LSMPhysicalParameters::m2 + 0.5 * LSMPhysicalParameters::lambda_quartic * rho;
    }

    void set_time(double t_)
    {
      this->t = t_;
      this->k = std::exp(-t_) * this->Lambda;
      flows_.set_k(this->k);
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F, const Point<dim> & /*x*/,
              const Solution &sol) const
    {
      const auto &fe_functions = get<"fe_functions">(sol);
      const NT m2Pi = fe_functions[0];
      NT pion_loop;
      flows_.V_pion.get(pion_loop, this->k, LSMPhysicalParameters::N, LSMPhysicalParameters::T, m2Pi);
      F[0][0] = LSMPhysicalParameters::N * pion_loop;
    }

  private:
    mutable PolyExpFlows flows_;
  };

  // --- CG model (matches Examples/ONfiniteT/model.hh) ------------------------
  //
  // CG uses ρ-coords with a combined flux (no advection/diffusion split,
  // because CG doesn't need the KT-style separation). Plus def::LLFFlux +
  // def::FlowBoundaries instead of KT's wave-speed/extrapolation machinery.

  class LSM_CG : public def::AbstractModel<LSM_CG, Components>,
                 public def::fRG,
                 public def::LLFFlux<LSM_CG>,
                 public def::FlowBoundaries<LSM_CG>,
                 public def::AD<LSM_CG>
  {
  public:
    LSM_CG(const ConfigTree &json, const GridSettings & /*grid*/) : def::fRG(json), flows_(json)
    {
      flows_.set_k(this->Lambda);
      flows_.set_T(LSMPhysicalParameters::T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double rho = x[0];
      values[0] = LSMPhysicalParameters::m2 + 0.5 * LSMPhysicalParameters::lambda_quartic * rho;
    }

    void set_time(double t_)
    {
      this->t = t_;
      this->k = std::exp(-t_) * this->Lambda;
      flows_.set_k(this->k);
    }

    // Combined CG flux: F = (N-1)·V_pion(m²_Pi) + V_sigma(m²_Sigma).
    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F, const Point<dim> &x,
              const Solution &sol) const
    {
      const auto &fe_functions = get<"fe_functions">(sol);
      const auto &fe_derivatives = get<"fe_derivatives">(sol);
      const NT u = fe_functions[0];
      const NT du_drho = fe_derivatives[0][0];
      const double rho = x[0];
      const NT m2Pi = u;
      const NT m2Sigma = u + NT(2.0 * rho) * du_drho;
      NT pion_loop, sigma_loop;
      flows_.V_pion.get(pion_loop, this->k, LSMPhysicalParameters::N, LSMPhysicalParameters::T, m2Pi);
      flows_.V_sigma.get(sigma_loop, this->k, LSMPhysicalParameters::N, LSMPhysicalParameters::T, m2Sigma);
      F[0][0] = (LSMPhysicalParameters::N - 1.0) * pion_loop + sigma_loop;
    }

  private:
    mutable PolyExpFlows flows_;
  };

} // namespace on_kt_3D
