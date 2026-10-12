#include <catch2/catch_all.hpp>

#include <DiFfRG/model/model.hh>
#include <DiFfRG/model/numflux.hh>

using namespace dealii;
using namespace DiFfRG;

namespace
{
  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>;

  /// Linear advection, F(u) = c u, with the LLF numerical flux.
  template <int dim> struct Advection : public def::AbstractModel<Advection<dim>, Components>,
                                        public def::LLFFlux<Advection<dim>> {
    Tensor<1, dim> c;

    template <int d, typename NT, typename Solution>
    void flux(std::array<Tensor<1, d, NT>, 1> &F, const Point<d> &, const Solution &sol) const
    {
      for (int i = 0; i < d; ++i)
        F[0][i] = c[i] * get<"fe_functions">(sol)[0];
    }
  };

  using Extractors = std::array<double, 0>;
  using Variables = Vector<double>;

  /// The per-point LLF numerical flux of @p model at a face with outward normal @p normal of the s side, dotted with
  /// @p normal.
  template <int dim>
  double llf(const Advection<dim> &model, const Tensor<1, dim> &normal, const double u_s, const double u_n)
  {
    const Extractors e{};
    const Variables v;
    std::array<double, 1> vs{{u_s}}, vn{{u_n}};
    double width = 1.;
    std::array<Tensor<1, dim>, 1> NF{};
    model.numflux(NF, normal, Point<dim>(), batch_tie<false, false>(vs, e, v, width),
                  batch_tie<false, false>(vn, e, v, width));
    return NF[0] * normal;
  }

  /// The same through the batched numflux_batch.
  template <int dim>
  double llf_batch(const Advection<dim> &model, const Tensor<1, dim> &normal, const double u_s, const double u_n)
  {
    using Batch = PointBatch<dim, double, 1, Extractors, Variables, false, false>;
    const Extractors e{};
    const Variables v;
    Batch s, n;
    for (auto *b : {&s, &n}) {
      b->reinit(1);
      b->set_shared(e, v);
      b->width(0) = 1.;
      for (int d = 0; d < dim; ++d)
        b->coordinate(d, 0) = 0.;
    }
    s.value(0, 0) = u_s;
    n.value(0, 0) = u_n;
    BatchOutput<dim, double, 1> out;
    out.reinit(1, Term::flux);
    const std::vector<Tensor<1, dim>> normals{normal};
    model.numflux_batch(out, normals, s, n);
    double result = 0.;
    for (int d = 0; d < dim; ++d)
      result += out.flux(0, d)[0] * normal[d];
    return result;
  }
} // namespace

TEMPLATE_TEST_CASE_SIG("LLF is the upwind flux for linear advection, for either normal direction", "[model][numflux]",
                       ((bool batched), batched), (false), (true))
{
  Advection<1> model;
  model.c[0] = 2.;
  const auto H = [&](const Tensor<1, 1> &n, double u_s, double u_n) {
    return batched ? llf_batch(model, n, u_s, u_n) : llf(model, n, u_s, u_n);
  };
  const double u_left = 3., u_right = 5.;
  // c > 0: the face flux is c u_left whichever side is s; dotted with the normal of s it changes sign.
  const Tensor<1, 1> right({1.}), left({-1.});
  CHECK(H(right, u_left, u_right) == Catch::Approx(2. * u_left).epsilon(1e-6));
  CHECK(H(left, u_right, u_left) == Catch::Approx(-2. * u_left).epsilon(1e-6));
}

TEMPLATE_TEST_CASE_SIG("LLF is conservative for any normal", "[model][numflux]", ((bool batched), batched), (false),
                       (true))
{
  Advection<2> model;
  model.c[0] = 1.5;
  model.c[1] = -0.7;
  const auto H = [&](const Tensor<1, 2> &n, double u_s, double u_n) {
    return batched ? llf_batch(model, n, u_s, u_n) : llf(model, n, u_s, u_n);
  };
  for (const auto &n : {Tensor<1, 2>({1., 0.}), Tensor<1, 2>({0., -1.}), Tensor<1, 2>({0.6, -0.8})}) {
    // What leaves s enters n.
    CHECK(H(n, 1.3, -0.4) == Catch::Approx(-H(-n, -0.4, 1.3)).epsilon(1e-10));
    // Rusanov: H = avg(F).n - |c.n|/2 (u_n - u_s), with the exact wave speed |c.n| up to the FD step.
    const double cn = model.c * n;
    CHECK(H(n, 1.3, -0.4) == Catch::Approx(0.5 * cn * (1.3 - 0.4) - 0.5 * std::abs(cn) * (-0.4 - 1.3)).epsilon(1e-6));
  }
}
