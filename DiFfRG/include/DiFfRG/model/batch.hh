#pragma once

// DiFfRG
#include <DiFfRG/common/tuples.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/model/ad.hh>
#include <DiFfRG/physics/integration/point_arg.hh>

// external libraries
#include <autodiff/forward/real.hpp>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <tbb/tbb.h>

// standard library
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief The per-point solution tuple the per-point model callbacks (flux, source, numflux, boundary_numflux and
   * their jacobians) receive from a batched FEM assembler. It holds only the inputs the assembler gathered:
   * - values only (DG, or batch_reads_derivatives = false): "fe_functions", "extractors", "variables", "cell_width"
   * - with derivatives (batch_reads_hessians = false): "fe_functions", "fe_derivatives", "extractors", ...
   * - with derivatives and hessians: "fe_functions", "fe_derivatives", "fe_hessians", "extractors", ...
   *
   * A model that reads an entry its assembler does not provide therefore fails to compile.
   */
  template <bool with_derivatives, bool with_hessians, typename... T> auto batch_tie(T &&...t)
  {
    static_assert(with_derivatives || !with_hessians, "Hessians are only gathered together with derivatives.");
    if constexpr (with_hessians)
      return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives", "fe_hessians", "extractors",
                                                       "variables", "cell_width">>(std::tie(t...));
    else if constexpr (with_derivatives)
      return named_tuple<std::tuple<T &...>,
                         StringSet<"fe_functions", "fe_derivatives", "extractors", "variables", "cell_width">>(
          std::tie(t...));
    else
      return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "extractors", "variables", "cell_width">>(
          std::tie(t...));
  }

  /**
   * @brief The terms a batch evaluation (Model::evaluate_batch) can produce. The assembler requests only
   * those it uses, e.g. `Term::flux | Term::source` at cell points and `Term::flux` at faces.
   */
  enum class Term : unsigned { flux = 1, source = 2, diffusion_flux = 4 };
  constexpr Term operator|(const Term a, const Term b) { return Term(unsigned(a) | unsigned(b)); }
  /// Whether the set of terms @p set contains @p t.
  constexpr bool contains(const Term set, const Term t) { return (unsigned(set) & unsigned(t)) != 0; }
  /// The terms a batch's per-point tuple can be evaluated for: Batch::terms if it declares them (the KT traces
  /// carry the tuple of one term only), otherwise all.
  template <typename Batch> constexpr Term batch_terms()
  {
    if constexpr (requires { Batch::terms; })
      return Batch::terms;
    else
      return Term::flux | Term::source | Term::diffusion_flux;
  }

  namespace internal
  {
    /// v = n copies of value, filled in parallel: these buffers hold up to millions of entries.
    template <typename T> void parallel_assign(std::vector<T> &v, const size_t n, const T &value)
    {
      v.resize(n);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, n, 1 << 14), [&](const tbb::blocked_range<size_t> &r) {
        std::fill(v.begin() + r.begin(), v.begin() + r.end(), value);
      });
    }
  } // namespace internal

  /// Copy of the solution at a single point of a PointBatch.
  template <int dim, typename NT, size_t n_fe> struct PointState {
    std::array<NT, n_fe> values;
    std::array<dealii::Tensor<1, dim, NT>, n_fe> derivatives;
    std::array<dealii::Tensor<2, dim, NT>, n_fe> hessians;
    double cell_width;
  };

  /**
   * @brief The FE solution at a set of points: values and (if gathered) derivatives and hessians of every
   * FE function, plus the position and cell width of each point. Extractors and variables are shared by
   * all points.
   *
   * Storage is component-major with the points fastest, so values(c), derivatives(c, d) and
   * hessians(c, d1, d2) are contiguous columns (PointSpan) that can be passed directly as per-point
   * arguments to an integrator's map_points().
   *
   * Point i means the same point in every column, and in every column of the output BatchOutput.
   * The order of the points carries no meaning for the model.
   *
   * @tparam with_derivatives, with_hessians which inputs the assembler gathers. Reading one it does not gather
   * (derivatives() / hessians(), or the matching entry of the per-point tuple) does not compile; a model
   * shared between assemblers asks `if constexpr (Batch::has_derivatives)`.
   */
  template <int dim_, typename NT, size_t n_fe, typename Extractors, typename Variables, bool with_derivatives,
            bool with_hessians>
  class PointBatch
  {
    static_assert(with_derivatives || !with_hessians, "Hessians are only gathered together with derivatives.");

  public:
    static constexpr int dim = dim_;
    static constexpr size_t n_fe_functions = n_fe;
    static constexpr bool has_derivatives = with_derivatives;
    static constexpr bool has_hessians = with_hessians;
    using number_type = NT;
    using extractors_type = Extractors;
    /// The same batch in another number type, e.g. for AD.
    template <typename NT2>
    using rebind = PointBatch<dim, NT2, n_fe, std::array<NT2, std::tuple_size_v<Extractors>>, Variables,
                              with_derivatives, with_hessians>;
    /// The copy of one point that the per-point callbacks see, see tie().
    using State = PointState<dim, NT, n_fe>;
    /// Where tie() puts the extractors, i.e. the tuple index of the per-point extractor jacobians.
    static constexpr uint extractor_index = 1 + with_derivatives + with_hessians;

    void reinit(const size_t n_points)
    {
      n = n_points;
      state.resize(n * n_columns);
      positions.resize(n * dim);
      widths.resize(n);
    }

    void set_shared(const Extractors &extractors, const Variables &variables)
    {
      m_extractors = &extractors;
      m_variables = &variables;
    }
    /// Forget the shared extractors and variables, e.g. when the call that owns them returns.
    void clear_shared()
    {
      m_extractors = nullptr;
      m_variables = nullptr;
    }

    size_t size() const { return n; }

    PointSpan<const NT> values(const size_t c) const { return {column(value_column(c)), n}; }
    /// Only if the assembler gathers derivatives (not under DG, nor with batch_reads_derivatives = false).
    PointSpan<const NT> derivatives(const size_t c, const size_t d) const
      requires with_derivatives
    {
      return {column(derivative_column(c, d)), n};
    }
    /// Only if the assembler gathers hessians (not under DG, nor with batch_reads_hessians = false).
    PointSpan<const NT> hessians(const size_t c, const size_t d1, const size_t d2) const
      requires with_hessians
    {
      return {column(hessian_column(c, d1, d2)), n};
    }
    PointSpan<const double> coordinates(const size_t d) const { return {positions.data() + d * n, n}; }

    dealii::Point<dim> x(const size_t i) const
    {
      dealii::Point<dim> p;
      for (int d = 0; d < dim; ++d)
        p[d] = positions[d * n + i];
      return p;
    }
    double cell_width(const size_t i) const { return widths[i]; }

    const Extractors &extractors() const { return *m_extractors; }
    const Variables &variables() const { return *m_variables; }

    /// Point i as a State; only the gathered inputs are loaded.
    void load(const size_t i, State &s) const
    {
      for (size_t c = 0; c < n_fe; ++c) {
        s.values[c] = column(value_column(c))[i];
        if constexpr (with_derivatives)
          for (int d1 = 0; d1 < dim; ++d1) {
            s.derivatives[c][d1] = column(derivative_column(c, d1))[i];
            if constexpr (with_hessians)
              for (int d2 = 0; d2 < dim; ++d2)
                s.hessians[c][d1][d2] = column(hessian_column(c, d1, d2))[i];
          }
      }
      s.cell_width = widths[i];
    }

    /// The named tuple the per-point callbacks receive for a loaded State; see batch_tie.
    auto tie(State &s) const
    {
      if constexpr (with_hessians)
        return batch_tie<true, true>(s.values, s.derivatives, s.hessians, extractors(), variables(), s.cell_width);
      else if constexpr (with_derivatives)
        return batch_tie<true, false>(s.values, s.derivatives, extractors(), variables(), s.cell_width);
      else
        return batch_tie<false, false>(s.values, extractors(), variables(), s.cell_width);
    }

    /// Point j of this batch = point i of @p src, which has the same inputs but may differ in number type.
    template <typename Src> void copy_point(const size_t j, const Src &src, const size_t i)
    {
      for (size_t c = 0; c < n_fe; ++c) {
        value(c, j) = src.values(c)[i];
        if constexpr (with_derivatives)
          for (int d1 = 0; d1 < dim; ++d1) {
            derivative(c, d1, j) = src.derivatives(c, d1)[i];
            if constexpr (with_hessians)
              for (int d2 = 0; d2 < dim; ++d2)
                hessian(c, d1, d2, j) = src.hessians(c, d1, d2)[i];
          }
      }
      for (int d = 0; d < dim; ++d)
        coordinate(d, j) = src.coordinates(d)[i];
      width(j) = src.cell_width(i);
    }

    // Writable access for whoever fills the batch.
    NT &value(const size_t c, const size_t i) { return column(value_column(c))[i]; }
    NT &derivative(const size_t c, const size_t d, const size_t i)
      requires with_derivatives
    {
      return column(derivative_column(c, d))[i];
    }
    NT &hessian(const size_t c, const size_t d1, const size_t d2, const size_t i)
      requires with_hessians
    {
      return column(hessian_column(c, d1, d2))[i];
    }
    double &coordinate(const size_t d, const size_t i) { return positions[d * n + i]; }
    double &width(const size_t i) { return widths[i]; }

  protected:
    const NT *column(const size_t col) const { return state.data() + col * n; }
    NT *column(const size_t col) { return state.data() + col * n; }

  private:
    static constexpr size_t n_columns = n_fe * (1 + (with_derivatives ? dim : 0) + (with_hessians ? dim * dim : 0));
    static constexpr size_t value_column(const size_t c) { return c; }
    static constexpr size_t derivative_column(const size_t c, const size_t d) { return n_fe + c * dim + d; }
    static constexpr size_t hessian_column(const size_t c, const size_t d1, const size_t d2)
    {
      return n_fe * (1 + dim) + (c * dim + d1) * dim + d2;
    }

    size_t n = 0;
    std::vector<NT> state;
    std::vector<double> positions;
    std::vector<double> widths;
    const Extractors *m_extractors = nullptr;
    const Variables *m_variables = nullptr;
  };

  namespace internal
  {
    /// Number of FE functions on all levels of an LDG model together.
    template <typename Components, size_t... L> constexpr size_t count_all_levels(std::index_sequence<L...>)
    {
      return (Components::count_fe_functions(L) + ...);
    }
    template <typename NT, typename Components, size_t... L>
    std::tuple<std::array<NT, Components::count_fe_functions(L)>...> level_arrays(std::index_sequence<L...>);

    /// The per-point tuple of an LDG main level with n_levels levels, named as the LDG assembler always did.
    template <uint n_levels, typename... T> auto ldg_tie(T &&...t)
    {
      if constexpr (n_levels == 2)
        return named_tuple<std::tuple<T &...>,
                           StringSet<"fe_functions", "LDG1", "extractors", "variables", "cell_width">>(std::tie(t...));
      else if constexpr (n_levels == 3)
        return named_tuple<std::tuple<T &...>,
                           StringSet<"fe_functions", "LDG1", "LDG2", "extractors", "variables", "cell_width">>(
            std::tie(t...));
      else {
        static_assert(n_levels == 4, "LDG supports at most three levels besides the FE functions.");
        return named_tuple<std::tuple<T &...>,
                           StringSet<"fe_functions", "LDG1", "LDG2", "LDG3", "extractors", "variables", "cell_width">>(
            std::tie(t...));
      }
    }
  } // namespace internal

  /**
   * @brief The solution at a set of points of an LDG main level: the values of the FE functions and of every
   * LDG level, plus position and cell width; no derivatives. `values(c)` is FE function c, as in PointBatch,
   * and `ldg_values(k, c)` component c of level k >= 1, both contiguous columns.
   *
   * Internally a PointBatch whose components are those of all levels in a row; seed-stacked AD therefore
   * differentiates with respect to all levels at once.
   */
  template <int dim_, typename NT, typename Components, typename Extractors, typename Variables>
  class LDGPointBatch : public PointBatch<dim_, NT,
                                          internal::count_all_levels<Components>(
                                              std::make_index_sequence<Components::count_fe_subsystems()>{}),
                                          Extractors, Variables, false, false>
  {
    using Base = PointBatch<
        dim_, NT, internal::count_all_levels<Components>(std::make_index_sequence<Components::count_fe_subsystems()>{}),
        Extractors, Variables, false, false>;

  public:
    static constexpr uint n_levels = Components::count_fe_subsystems();
    static constexpr uint extractor_index = n_levels;
    template <typename NT2>
    using rebind = LDGPointBatch<dim_, NT2, Components, std::array<NT2, std::tuple_size_v<Extractors>>, Variables>;

    /// Column of component 0 of level k among the components of all levels.
    static constexpr size_t level_offset(const uint k)
    {
      size_t offset = 0;
      for (uint l = 0; l < k; ++l)
        offset += Components::count_fe_functions(l);
      return offset;
    }

    PointSpan<const NT> ldg_values(const uint k, const size_t c) const { return Base::values(level_offset(k) + c); }
    NT &ldg_value(const uint k, const size_t c, const size_t i) { return Base::value(level_offset(k) + c, i); }

    struct State {
      decltype(internal::level_arrays<NT, Components>(std::make_index_sequence<n_levels>{})) levels;
      double cell_width;
    };

    void load(const size_t i, State &s) const
    {
      constexpr_for<0, n_levels, 1>([&](auto l) {
        auto &level = std::get<l>(s.levels);
        for (size_t c = 0; c < level.size(); ++c)
          level[c] = this->column(level_offset(l) + c)[i]; // not Base::, GCC 13 ICEs on it
      });
      s.cell_width = this->cell_width(i);
    }

    auto tie(State &s) const
    {
      return std::apply(
          [&](auto &...level) {
            return internal::ldg_tie<n_levels>(level..., this->extractors(), this->variables(), s.cell_width);
          },
          s.levels);
    }
  };

  /**
   * @brief The requested terms (flux, source, diffusion flux) of every FE function at the points of a
   * PointBatch, as contiguous columns (PointSpan). Only requested terms are stored, and asking for another one
   * throws; reinit() zeroes them, so a model only writes what it computes.
   */
  template <int dim, typename NT, size_t n_fe> class BatchOutput
  {
  public:
    static constexpr size_t n_components = n_fe;

    void reinit(const size_t n_points, const Term terms)
    {
      n = n_points;
      m_terms = terms;
      size_t columns = 0;
      const auto place = [&](const Term t, const size_t width) {
        if (!contains(terms, t)) return no_column;
        const size_t offset = columns;
        columns += width;
        return offset;
      };
      flux_offset = place(Term::flux, n_fe * dim);
      source_offset = place(Term::source, n_fe);
      diffusion_offset = place(Term::diffusion_flux, n_fe * dim);
      internal::parallel_assign(data, n * columns, NT(0));
    }
    size_t size() const { return n; }
    /// Whether the assembler asked for @p t; a model may skip computing anything else.
    bool requested(const Term t) const { return contains(m_terms, t); }

    PointSpan<NT> flux(const size_t c, const size_t d) { return column(flux_offset, c * dim + d, "flux"); }
    PointSpan<const NT> flux(const size_t c, const size_t d) const { return column(flux_offset, c * dim + d, "flux"); }
    PointSpan<NT> source(const size_t c) { return column(source_offset, c, "source"); }
    PointSpan<const NT> source(const size_t c) const { return column(source_offset, c, "source"); }
    PointSpan<NT> diffusion_flux(const size_t c, const size_t d)
    {
      return column(diffusion_offset, c * dim + d, "diffusion flux");
    }
    PointSpan<const NT> diffusion_flux(const size_t c, const size_t d) const
    {
      return column(diffusion_offset, c * dim + d, "diffusion flux");
    }

    void store_flux(const size_t i, const std::array<dealii::Tensor<1, dim, NT>, n_fe> &F)
    {
      for (size_t c = 0; c < n_fe; ++c)
        for (int d = 0; d < dim; ++d)
          flux(c, d)[i] = F[c][d];
    }
    void store_source(const size_t i, const std::array<NT, n_fe> &S)
    {
      for (size_t c = 0; c < n_fe; ++c)
        source(c)[i] = S[c];
    }
    void store_diffusion_flux(const size_t i, const std::array<dealii::Tensor<1, dim, NT>, n_fe> &D)
    {
      for (size_t c = 0; c < n_fe; ++c)
        for (int d = 0; d < dim; ++d)
          diffusion_flux(c, d)[i] = D[c][d];
    }

  private:
    static constexpr size_t no_column = size_t(-1);
    PointSpan<NT> column(const size_t offset, const size_t col, const char *term)
    {
      const auto c = std::as_const(*this).column(offset, col, term);
      return {const_cast<NT *>(c.data()), c.size()};
    }
    PointSpan<const NT> column(const size_t offset, const size_t col, const char *term) const
    {
      if (offset == no_column)
        throw std::logic_error(std::string("BatchOutput: the ") + term + " was not requested by the assembler.");
      return {data.data() + (offset + col) * n, n};
    }

    size_t n = 0;
    Term m_terms = Term::flux;
    size_t flux_offset = no_column, source_offset = no_column, diffusion_offset = no_column;
    std::vector<NT> data;
  };

  /**
   * @brief Derivatives of the n_out components of flux and source at one point with respect to the n_in input
   * components (values, derivatives, hessians) and the extractors. Same layout as the per-point
   * jacobian_flux_source* callbacks. At face points the flux blocks hold the numerical flux and the source
   * blocks stay zero; an interior face point has one of these per trace, and the extractor blocks of trace 0
   * hold the total.
   */
  template <int dim, size_t n_out, size_t n_in, size_t n_extr> struct PointJacobian {
    using T1 = dealii::Tensor<1, dim, double>;
    SimpleMatrix<T1, n_out, n_in> j_flux;
    SimpleMatrix<dealii::Tensor<1, dim, T1>, n_out, n_in> j_grad_flux;
    SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<2, dim, double>>, n_out, n_in> j_hess_flux;
    SimpleMatrix<T1, n_out, n_extr> j_extr_flux;
    SimpleMatrix<double, n_out, n_in> j_source;
    SimpleMatrix<T1, n_out, n_in> j_grad_source;
    SimpleMatrix<dealii::Tensor<2, dim, double>, n_out, n_in> j_hess_source;
    SimpleMatrix<double, n_out, n_extr> j_extr_source;

    /// Whether the value, source and extractor blocks are finite (their tensors entry by entry).
    bool is_finite() const
    {
      const auto finite = [](const auto &m, const uint rows, const uint cols) {
        for (uint r = 0; r < rows; ++r)
          for (uint c = 0; c < cols; ++c)
            if (!std::isfinite(double(m(r, c) * m(r, c)))) return false;
        return true;
      };
      return finite(j_flux, n_out, n_in) && finite(j_source, n_out, n_in) && finite(j_extr_flux, n_out, n_extr) &&
             finite(j_extr_source, n_out, n_extr);
    }
  };
  /// The jacobian at an interior face point: one block set per trace.
  template <int dim, size_t n_out, size_t n_in, size_t n_extr>
  using FaceJacobian = std::array<PointJacobian<dim, n_out, n_in, n_extr>, 2>;

  /**
   * @brief Buffers of the seed-stacked AD jacobian (internal::seed_stacked_jacobian) of an evaluation with n_out
   * output components over batches of type Batch, kept by the caller across calls: they hold n_seeds copies of
   * the batch (of each of its n_sides traces), and allocating them anew every time costs as much as the
   * evaluation itself.
   */
  template <typename Batch, size_t n_out, size_t n_sides = 1> struct SeedStackWorkspace {
    using ADBatch = typename Batch::template rebind<autodiff::real>;
    static constexpr size_t n_extr = std::tuple_size_v<typename Batch::extractors_type>;
    std::array<ADBatch, n_sides> batches;
    BatchOutput<Batch::dim, autodiff::real, n_out> out;
    /// AD numbers, unless the batch freezes its extractors (rebind keeps them double).
    typename ADBatch::extractors_type extractors;
    std::vector<dealii::Tensor<1, Batch::dim>> normals;
  };

  /**
   * @brief Model traits read by the batched assemblers. A model declares
   * `static constexpr bool batch_reads_derivatives = false` (or `batch_reads_hessians`) when neither its
   * flux nor its source reads that input: the batch then skips it, and so do the AD jacobian seeds.
   */
  template <typename Model> constexpr bool batch_reads_derivatives()
  {
    if constexpr (requires { Model::batch_reads_derivatives; })
      return Model::batch_reads_derivatives;
    else
      return true;
  }
  template <typename Model> constexpr bool batch_reads_hessians()
  {
    if constexpr (requires { Model::batch_reads_hessians; })
      return Model::batch_reads_hessians;
    else
      return true;
  }

  namespace internal
  {
    /// f(i, x, state) at every point i of the batch, in a flat parallel loop; state is the loaded Batch::State.
    template <typename Batch, typename F> void for_each_state(const Batch &batch, const F &f)
    {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, batch.size()), [&](const tbb::blocked_range<size_t> &r) {
        typename Batch::State s;
        for (size_t i = r.begin(); i != r.end(); ++i) {
          batch.load(i, s);
          f(i, batch.x(i), s);
        }
      });
    }

    /// f(i, x, sol) at every point i of the batch, in a flat parallel loop; sol is the point's solution tuple.
    template <typename Batch, typename F> void for_each_point(const Batch &batch, const F &f)
    {
      for_each_state(batch, [&](const size_t i, const auto &x, auto &s) { f(i, x, batch.tie(s)); });
    }

    /// f(i, x, state_s, state_n) at every point i of a pair of trace batches, in a flat parallel loop.
    template <typename Batch, typename F>
    void for_each_state_pair(const Batch &batch_s, const Batch &batch_n, const F &f)
    {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, batch_s.size()), [&](const tbb::blocked_range<size_t> &r) {
        typename Batch::State s, n;
        for (size_t i = r.begin(); i != r.end(); ++i) {
          batch_s.load(i, s);
          batch_n.load(i, n);
          f(i, batch_s.x(i), s, n);
        }
      });
    }

    /// f(i, x, sol_s, sol_n) at every point i of a pair of trace batches, with the points' solution tuples.
    template <typename Batch, typename F>
    void for_each_point_pair(const Batch &batch_s, const Batch &batch_n, const F &f)
    {
      for_each_state_pair(batch_s, batch_n, [&](const size_t i, const auto &x, auto &s, auto &n) {
        f(i, x, batch_s.tie(s), batch_n.tie(n));
      });
    }

    /**
     * @brief The requested terms at all points of a batch from the model's per-point flux, source and
     * diffusion_flux. This is def::AbstractModel::evaluate_batch's default.
     *
     * The per-point callbacks run on several threads at once; a model whose per-point flux is not thread
     * safe (e.g. it calls a GPU integrator's get()) must override evaluate_batch.
     */
    template <typename Model, typename Out, typename Batch>
    void evaluate_per_point(const Model &model, Out &out, const Batch &batch)
    {
      using NT = typename Batch::number_type;
      constexpr int dim = Batch::dim;
      constexpr size_t n_out = Out::n_components;
      constexpr Term available = batch_terms<Batch>();
      for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        if constexpr (contains(available, Term::flux))
          if (out.requested(Term::flux)) {
            std::array<dealii::Tensor<1, dim, NT>, n_out> F{};
            model.flux(F, x, sol);
            out.store_flux(i, F);
          }
        if constexpr (contains(available, Term::source))
          if (out.requested(Term::source)) {
            std::array<NT, n_out> S{};
            model.source(S, x, sol);
            out.store_source(i, S);
          }
        if constexpr (contains(available, Term::diffusion_flux))
          if (out.requested(Term::diffusion_flux)) {
            std::array<dealii::Tensor<1, dim, NT>, n_out> D{};
            model.diffusion_flux(D, x, sol);
            out.store_diffusion_flux(i, D);
          }
      });
    }

    /**
     * @brief Level `dependent` of an LDG model at all points of a batch of level dependent - 1, from the model's
     * per-point ldg_flux and ldg_source. This is def::AbstractModel::ldg_evaluate_batch's default.
     */
    template <uint dependent, typename Model, typename Out, typename Batch>
    void ldg_evaluate_per_point(const Model &model, Out &out, const Batch &batch)
    {
      using NT = typename Batch::number_type;
      constexpr size_t n_out = Out::n_components;
      for_each_state(batch, [&](const size_t i, const auto &x, const auto &s) {
        if (out.requested(Term::flux)) {
          std::array<dealii::Tensor<1, Batch::dim, NT>, n_out> F{};
          model.template ldg_flux<dependent>(F, x, s.values);
          out.store_flux(i, F);
        }
        if (out.requested(Term::source)) {
          std::array<NT, n_out> S{};
          model.template ldg_source<dependent>(S, x, s.values);
          out.store_source(i, S);
        }
      });
    }
  } // namespace internal

  /**
   * @brief Boundary numflux at all points of a boundary batch, into the flux columns of @p out: the
   * model's `boundary_numflux_batch(out, normals, batch)` if it has one (def::FlowBoundaries does),
   * otherwise its per-point boundary_numflux. normals[i] is the outward normal at point i.
   */
  template <typename Model, typename Out, typename Batch>
  void evaluate_boundary_numflux(const Model &model, Out &out,
                                 const std::vector<dealii::Tensor<1, Batch::dim>> &normals, const Batch &batch)
  {
    if constexpr (requires { model.boundary_numflux_batch(out, normals, batch); })
      model.boundary_numflux_batch(out, normals, batch);
    else {
      using NT = typename Batch::number_type;
      internal::for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        std::array<dealii::Tensor<1, Batch::dim, NT>, Out::n_components> F{};
        model.boundary_numflux(F, normals[i], x, sol);
        out.store_flux(i, F);
      });
    }
  }

  /**
   * @brief Numerical flux at all points of an interior-face batch, into the flux columns of @p out: the
   * model's `numflux_batch(out, normals, batch_s, batch_n)` if it has one (def::LLFFlux does), otherwise its
   * per-point numflux. Point i of batch_s and batch_n are the two traces at the same face point, and
   * normals[i] points out of the cell of batch_s.
   */
  template <typename Model, typename Out, typename Batch>
  void evaluate_numflux(const Model &model, Out &out, const std::vector<dealii::Tensor<1, Batch::dim>> &normals,
                        const Batch &batch_s, const Batch &batch_n)
  {
    if constexpr (requires { model.numflux_batch(out, normals, batch_s, batch_n); })
      model.numflux_batch(out, normals, batch_s, batch_n);
    else {
      using NT = typename Batch::number_type;
      internal::for_each_point_pair(batch_s, batch_n,
                                    [&](const size_t i, const auto &x, const auto &sol_s, const auto &sol_n) {
                                      std::array<dealii::Tensor<1, Batch::dim, NT>, Out::n_components> F{};
                                      model.numflux(F, normals[i], x, sol_s, sol_n);
                                      out.store_flux(i, F);
                                    });
    }
  }

  /**
   * @brief Level `to` of an LDG model at all points of a batch of level to - 1: flux and/or source, as @p out
   * requests. See Model::ldg_evaluate_batch.
   */
  template <uint to, typename Model, typename Out, typename Batch>
  void evaluate_ldg_level(const Model &model, Out &out, const Batch &batch)
  {
    model.template ldg_evaluate_batch<to>(out, batch);
  }

  /// The boundary numflux of LDG level `to`: the model's ldg_boundary_numflux_batch, else its per-point one.
  template <uint to, typename Model, typename Out, typename Batch>
  void evaluate_ldg_boundary_numflux(const Model &model, Out &out,
                                     const std::vector<dealii::Tensor<1, Batch::dim>> &normals, const Batch &batch)
  {
    if constexpr (requires { model.template ldg_boundary_numflux_batch<to>(out, normals, batch); })
      model.template ldg_boundary_numflux_batch<to>(out, normals, batch);
    else {
      using NT = typename Batch::number_type;
      internal::for_each_state(batch, [&](const size_t i, const auto &x, const auto &s) {
        std::array<dealii::Tensor<1, Batch::dim, NT>, Out::n_components> F{};
        model.template ldg_boundary_numflux<to>(F, normals[i], x, s.values);
        out.store_flux(i, F);
      });
    }
  }

  /// The numerical flux of LDG level `to`: the model's ldg_numflux_batch (def::LDGUpDownFluxes has one), else
  /// its per-point ldg_numflux.
  template <uint to, typename Model, typename Out, typename Batch>
  void evaluate_ldg_numflux(const Model &model, Out &out, const std::vector<dealii::Tensor<1, Batch::dim>> &normals,
                            const Batch &batch_s, const Batch &batch_n)
  {
    if constexpr (requires { model.template ldg_numflux_batch<to>(out, normals, batch_s, batch_n); })
      model.template ldg_numflux_batch<to>(out, normals, batch_s, batch_n);
    else {
      using NT = typename Batch::number_type;
      internal::for_each_state_pair(batch_s, batch_n, [&](const size_t i, const auto &x, const auto &s, const auto &n) {
        std::array<dealii::Tensor<1, Batch::dim, NT>, Out::n_components> F{};
        model.template ldg_numflux<to>(F, normals[i], x, s.values, n.values);
        out.store_flux(i, F);
      });
    }
  }

  namespace internal
  {
    /// One forward-mode AD seed direction among the FE inputs of one trace of a point.
    struct BatchSeed {
      enum Kind { value, derivative, hessian } kind;
      uint c, d1, d2;
      uint side;
    };

    // Forward-mode AD on autodiff::real: component 1 carries the derivative along the seeded direction.
    inline void seed_direction(autodiff::real &x) { x[1] = 1.; }
    inline void clear_direction(autodiff::real &x) { x[1] = 0.; }
    inline double along_direction(const autodiff::real &x) { return x[1]; }

    template <int dim, size_t n_in>
    std::vector<BatchSeed> batch_seeds(const uint n_sides, const bool with_derivatives, const bool with_hessians)
    {
      std::vector<BatchSeed> seeds;
      for (uint side = 0; side < n_sides; ++side) {
        for (uint c = 0; c < n_in; ++c)
          seeds.push_back({BatchSeed::value, c, 0, 0, side});
        if (with_derivatives)
          for (uint c = 0; c < n_in; ++c)
            for (uint d = 0; d < dim; ++d)
              seeds.push_back({BatchSeed::derivative, c, d, 0, side});
        // A hessian is symmetric, so (d1, d2) and (d2, d1) are seeded together.
        if (with_hessians)
          for (uint c = 0; c < n_in; ++c)
            for (uint d1 = 0; d1 < dim; ++d1)
              for (uint d2 = d1; d2 < dim; ++d2)
                seeds.push_back({BatchSeed::hessian, c, d1, d2, side});
      }
      return seeds;
    }

    /**
     * @brief Forward-mode AD jacobian of a batch evaluation, with all seed directions stacked along
     * the point axis: block b of the AD batches is a copy of the input batches with seed b set at every
     * point, so a single evaluation yields the jacobian column of seed b at all points. Seeds are
     * stacked up to max_stacked points per evaluation. Extractor seeds are one evaluation each, as the
     * extractors are shared by all points (and traces) of a batch; their total derivative goes to side 0.
     * Derivatives and hessians are seeded if the batches hold them.
     *
     * @tparam n_out the number of output components of the evaluation
     * @param evaluate callable(BatchOutput<AD> &out, const std::array<ADBatch, n_sides> &batches, size_t n_blocks)
     * @param jacobian_at callable(i, side) -> PointJacobian & of point i with respect to trace `side`
     * @param batches the n_sides traces of the points (one for cells and boundary faces, two for interior faces)
     * @param terms the terms evaluate produces; only those are read
     * @param extractor_seeds whether to differentiate with respect to the extractors, too; must be false for a
     * batch that freezes its extractors (Batch::rebind keeps them double, e.g. the KT TraceBatch)
     */
    template <size_t n_out, size_t n_sides, typename Evaluate, typename JacobianAt, typename Batch>
    void seed_stacked_jacobian(const Evaluate &evaluate, const JacobianAt &jacobian_at,
                               const std::array<const Batch *, n_sides> &batches, const size_t max_stacked,
                               const Term terms, SeedStackWorkspace<Batch, n_out, n_sides> &workspace,
                               const bool extractor_seeds = true)
    {
      constexpr int dim = Batch::dim;
      constexpr size_t n_in = Batch::n_fe_functions;
      constexpr size_t n_extr = SeedStackWorkspace<Batch, n_out, n_sides>::n_extr;
      const size_t n = batches[0]->size();
      if (n == 0) return;
      const bool with_flux = contains(terms, Term::flux), with_source = contains(terms, Term::source);

      auto &ad_extractors = workspace.extractors;
      for (size_t e = 0; e < n_extr; ++e)
        ad_extractors[e] = batches[0]->extractors()[e];

      auto &ad_batches = workspace.batches;
      auto &ad_out = workspace.out;
      const auto prepare = [&](const size_t n_blocks) {
        for (size_t s = 0; s < n_sides; ++s) {
          ad_batches[s].reinit(n_blocks * n);
          ad_batches[s].set_shared(ad_extractors, batches[s]->variables());
        }
        ad_out.reinit(n_blocks * n, terms);
      };
      // Slot b * n + i of every AD trace := point i of that trace, unseeded.
      const auto copy_block_point = [&](const size_t b, const size_t i) {
        for (size_t s = 0; s < n_sides; ++s)
          ad_batches[s].copy_point(b * n + i, *batches[s], i);
      };

      const auto seeds = batch_seeds<dim, n_in>(n_sides, Batch::has_derivatives, Batch::has_hessians);
      const size_t per_group = std::max<size_t>(1, max_stacked / n);
      for (size_t g0 = 0; g0 < seeds.size(); g0 += per_group) {
        const size_t G = std::min(per_group, seeds.size() - g0);
        prepare(G);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            for (size_t b = 0; b < G; ++b) {
              copy_block_point(b, i);
              const auto &s = seeds[g0 + b];
              auto &ad = ad_batches[s.side];
              const size_t j = b * n + i;
              // batch_seeds() only lists the derivatives and hessians a batch holds.
              if (s.kind == BatchSeed::value)
                seed_direction(ad.value(s.c, j));
              else if constexpr (Batch::has_derivatives) {
                if (s.kind == BatchSeed::derivative)
                  seed_direction(ad.derivative(s.c, s.d1, j));
                else if constexpr (Batch::has_hessians) {
                  seed_direction(ad.hessian(s.c, s.d1, s.d2, j));
                  if (s.d1 != s.d2) seed_direction(ad.hessian(s.c, s.d2, s.d1, j));
                }
              }
            }
        });

        evaluate(ad_out, std::as_const(ad_batches), G);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            for (size_t b = 0; b < G; ++b) {
              const auto &s = seeds[g0 + b];
              const size_t j = b * n + i;
              auto &Ji = jacobian_at(i, s.side);
              for (uint ci = 0; ci < n_out; ++ci) {
                if (with_flux)
                  for (int d = 0; d < dim; ++d) {
                    const double gf = along_direction(ad_out.flux(ci, d)[j]);
                    if (s.kind == BatchSeed::value)
                      Ji.j_flux(ci, s.c)[d] = gf;
                    else if (s.kind == BatchSeed::derivative)
                      Ji.j_grad_flux(ci, s.c)[d][s.d1] = gf;
                    else if (s.d1 == s.d2)
                      Ji.j_hess_flux(ci, s.c)[d][s.d1][s.d1] = gf;
                    else {
                      // The joint seed yields dF/dH_12 + dF/dH_21; only the symmetric part is contracted.
                      Ji.j_hess_flux(ci, s.c)[d][s.d1][s.d2] = gf / 2;
                      Ji.j_hess_flux(ci, s.c)[d][s.d2][s.d1] = gf / 2;
                    }
                  }
                if (with_source) {
                  const double gs = along_direction(ad_out.source(ci)[j]);
                  if (s.kind == BatchSeed::value)
                    Ji.j_source(ci, s.c) = gs;
                  else if (s.kind == BatchSeed::derivative)
                    Ji.j_grad_source(ci, s.c)[s.d1] = gs;
                  else if (s.d1 == s.d2)
                    Ji.j_hess_source(ci, s.c)[s.d1][s.d1] = gs;
                  else {
                    Ji.j_hess_source(ci, s.c)[s.d1][s.d2] = gs / 2;
                    Ji.j_hess_source(ci, s.c)[s.d2][s.d1] = gs / 2;
                  }
                }
              }
            }
        });
      }

      constexpr bool seedable_extractors =
          std::is_same_v<std::remove_cvref_t<decltype(ad_extractors[0])>, autodiff::real>;
      if (extractor_seeds && n_extr > 0 && !seedable_extractors)
        throw std::logic_error("seed_stacked_jacobian: the batch freezes its extractors; they cannot be seeded.");
      if constexpr (seedable_extractors)
        for (size_t e = 0; extractor_seeds && e < n_extr; ++e) {
          prepare(1);
          tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); ++i)
              copy_block_point(0, i);
          });
          seed_direction(ad_extractors[e]);
          evaluate(ad_out, std::as_const(ad_batches), 1);
          clear_direction(ad_extractors[e]);
          tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
              auto &Ji = jacobian_at(i, 0);
              for (uint ci = 0; ci < n_out; ++ci) {
                if (with_flux)
                  for (int d = 0; d < dim; ++d)
                    Ji.j_extr_flux(ci, e)[d] = along_direction(ad_out.flux(ci, d)[i]);
                if (with_source) Ji.j_extr_source(ci, e) = along_direction(ad_out.source(ci)[i]);
              }
            }
          });
        }
    }

    /// n_blocks copies of @p normals, one per block of a seed-stacked AD batch.
    template <typename Normal>
    const std::vector<Normal> &stack_normals(std::vector<Normal> &stacked, const std::vector<Normal> &normals,
                                             const size_t n_blocks)
    {
      stacked.resize(n_blocks * normals.size());
      for (size_t b = 0; b < n_blocks; ++b)
        std::copy(normals.begin(), normals.end(), stacked.begin() + b * normals.size());
      return stacked;
    }

    /// Whether the model's flux, source and numflux jacobians are the seed-stacked AD of its batch evaluation.
    template <typename Model> constexpr bool has_batch_ad_jacobians = std::is_base_of_v<def::BatchADJacobians, Model>;
    /// Whether those jacobians include the extractor directions (not for def::FE_AD).
    template <typename Model> constexpr bool seeds_extractors = !std::is_base_of_v<def::FrozenExtractors, Model>;

    template <typename Batch> constexpr bool is_ldg_batch = requires { Batch::n_levels; };

    /// Copy @p src, a jacobian block by the components of LDG level k, into the columns of level k of @p dst.
    template <typename Batch, uint k, typename Dst, typename NT, uint n_rows, uint n_columns>
    void copy_level_columns(Dst &dst, const SimpleMatrix<NT, n_rows, n_columns> &src)
    {
      for (uint r = 0; r < n_rows; ++r)
        for (uint c = 0; c < n_columns; ++c)
          dst(r, Batch::level_offset(k) + c) = src(r, c);
    }
  } // namespace internal

  /**
   * @brief Jacobian of flux and source at all points of a batch: for a model with AD jacobians
   * (def::AD) the seed-stacked AD of evaluate_batch, see internal::seed_stacked_jacobian; otherwise
   * the model's per-point jacobian_flux_source* callbacks.
   */
  template <typename Model, typename Batch, size_t n_out, size_t n_extr>
  void evaluate_flux_source_jacobian(const Model &model,
                                     std::vector<PointJacobian<Batch::dim, n_out, Batch::n_fe_functions, n_extr>> &J,
                                     const Batch &batch, const size_t max_stacked,
                                     SeedStackWorkspace<Batch, n_out> &workspace)
  {
    constexpr int dim = Batch::dim;
    internal::parallel_assign(J, batch.size(), {});
    if constexpr (internal::has_batch_ad_jacobians<Model>)
      internal::seed_stacked_jacobian<n_out, 1>(
          [&](auto &out, const auto &ad, size_t) { model.evaluate_batch(out, ad[0]); },
          [&](const size_t i, uint) -> auto & { return J[i]; }, std::array{&batch}, max_stacked,
          Term::flux | Term::source, workspace, internal::seeds_extractors<Model>);
    else
      internal::for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        auto &Ji = J[i];
        if constexpr (internal::is_ldg_batch<Batch>) {
          // One per-point call per level, as the level is a template parameter of the callback.
          constexpr_for<0, Batch::n_levels, 1>([&](auto k) {
            SimpleMatrix<dealii::Tensor<1, dim>, n_out, Model::Components::count_fe_functions(k)> jF{};
            SimpleMatrix<double, n_out, Model::Components::count_fe_functions(k)> jS{};
            model.template jacobian_flux_source<k, 0>(jF, jS, x, sol);
            internal::copy_level_columns<Batch, k>(Ji.j_flux, jF);
            internal::copy_level_columns<Batch, k>(Ji.j_source, jS);
          });
        } else {
          model.template jacobian_flux_source<0, 0>(Ji.j_flux, Ji.j_source, x, sol);
          model.template jacobian_flux_source_grad<1>(Ji.j_grad_flux, Ji.j_grad_source, x, sol);
          model.template jacobian_flux_source_hess<2>(Ji.j_hess_flux, Ji.j_hess_source, x, sol);
        }
        if constexpr (n_extr > 0)
          model.template jacobian_flux_source_extr<Batch::extractor_index>(Ji.j_extr_flux, Ji.j_extr_source, x, sol);
      });
  }

  /**
   * @brief Jacobian of the boundary numflux at all points of a boundary batch, into the flux blocks of
   * J. Same dispatch as evaluate_flux_source_jacobian, with the per-point jacobian_boundary_numflux*.
   */
  template <typename Model, typename Batch, size_t n_out, size_t n_extr>
  void
  evaluate_boundary_numflux_jacobian(const Model &model,
                                     std::vector<PointJacobian<Batch::dim, n_out, Batch::n_fe_functions, n_extr>> &J,
                                     const std::vector<dealii::Tensor<1, Batch::dim>> &normals, const Batch &batch,
                                     const size_t max_stacked, SeedStackWorkspace<Batch, n_out> &workspace)
  {
    constexpr int dim = Batch::dim;
    internal::parallel_assign(J, batch.size(), {});
    if constexpr (internal::has_batch_ad_jacobians<Model>)
      internal::seed_stacked_jacobian<n_out, 1>(
          [&](auto &out, const auto &ad, const size_t n_blocks) {
            evaluate_boundary_numflux(model, out, internal::stack_normals(workspace.normals, normals, n_blocks), ad[0]);
          },
          [&](const size_t i, uint) -> auto & { return J[i]; }, std::array{&batch}, max_stacked, Term::flux, workspace,
          internal::seeds_extractors<Model>);
    else
      internal::for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        auto &Ji = J[i];
        if constexpr (internal::is_ldg_batch<Batch>) {
          constexpr_for<0, Batch::n_levels, 1>([&](auto k) {
            SimpleMatrix<dealii::Tensor<1, dim>, n_out, Model::Components::count_fe_functions(k)> jF{};
            model.template jacobian_boundary_numflux<k, 0>(jF, normals[i], x, sol);
            internal::copy_level_columns<Batch, k>(Ji.j_flux, jF);
          });
        } else {
          model.template jacobian_boundary_numflux<0, 0>(Ji.j_flux, normals[i], x, sol);
          model.template jacobian_boundary_numflux_grad<1>(Ji.j_grad_flux, normals[i], x, sol);
          model.template jacobian_boundary_numflux_hess<2>(Ji.j_hess_flux, normals[i], x, sol);
        }
        if constexpr (n_extr > 0)
          model.template jacobian_boundary_numflux_extr<Batch::extractor_index>(Ji.j_extr_flux, normals[i], x, sol);
      });
  }

  /**
   * @brief Jacobian of the numerical flux at all points of an interior-face batch with respect to both
   * traces, into the flux blocks of J[i][0] (trace s) and J[i][1] (trace n); the extractor blocks of
   * J[i][0] hold the total. Same dispatch as evaluate_flux_source_jacobian, with the per-point
   * jacobian_numflux*.
   */
  template <typename Model, typename Batch, size_t n_out, size_t n_extr>
  void evaluate_numflux_jacobian(const Model &model,
                                 std::vector<FaceJacobian<Batch::dim, n_out, Batch::n_fe_functions, n_extr>> &J,
                                 const std::vector<dealii::Tensor<1, Batch::dim>> &normals, const Batch &batch_s,
                                 const Batch &batch_n, const size_t max_stacked,
                                 SeedStackWorkspace<Batch, n_out, 2> &workspace)
  {
    constexpr int dim = Batch::dim;
    constexpr size_t n_in = Batch::n_fe_functions;
    using T1 = dealii::Tensor<1, dim>;
    internal::parallel_assign(J, batch_s.size(), {});
    if constexpr (internal::has_batch_ad_jacobians<Model>)
      internal::seed_stacked_jacobian<n_out, 2>(
          [&](auto &out, const auto &ad, const size_t n_blocks) {
            evaluate_numflux(model, out, internal::stack_normals(workspace.normals, normals, n_blocks), ad[0], ad[1]);
          },
          [&](const size_t i, const uint side) -> auto & { return J[i][side]; }, std::array{&batch_s, &batch_n},
          max_stacked, Term::flux, workspace, internal::seeds_extractors<Model>);
    else
      internal::for_each_point_pair(
          batch_s, batch_n, [&](const size_t i, const auto &x, const auto &sol_s, const auto &sol_n) {
            if constexpr (internal::is_ldg_batch<Batch>) {
              constexpr_for<0, Batch::n_levels, 1>([&](auto k) {
                std::array<SimpleMatrix<T1, n_out, Model::Components::count_fe_functions(k)>, 2> jF{};
                model.template jacobian_numflux<k, 0>(jF, normals[i], x, sol_s, sol_n);
                for (uint side = 0; side < 2; ++side)
                  internal::copy_level_columns<Batch, k>(J[i][side].j_flux, jF[side]);
              });
            } else {
              std::array<SimpleMatrix<T1, n_out, n_in>, 2> j_value;
              std::array<SimpleMatrix<dealii::Tensor<1, dim, T1>, n_out, n_in>, 2> j_grad;
              std::array<SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<2, dim>>, n_out, n_in>, 2> j_hess;
              model.template jacobian_numflux<0, 0>(j_value, normals[i], x, sol_s, sol_n);
              model.template jacobian_numflux_grad<1>(j_grad, normals[i], x, sol_s, sol_n);
              model.template jacobian_numflux_hess<2>(j_hess, normals[i], x, sol_s, sol_n);
              for (uint side = 0; side < 2; ++side) {
                J[i][side].j_flux = j_value[side];
                J[i][side].j_grad_flux = j_grad[side];
                J[i][side].j_hess_flux = j_hess[side];
              }
            }
            if constexpr (n_extr > 0) {
              std::array<SimpleMatrix<T1, n_out, n_extr>, 2> j_extr;
              model.template jacobian_numflux_extr<Batch::extractor_index>(j_extr, normals[i], x, sol_s, sol_n);
              for (uint c = 0; c < n_out; ++c)
                for (uint e = 0; e < n_extr; ++e)
                  J[i][0].j_extr_flux(c, e) = j_extr[0](c, e) + j_extr[1](c, e);
            }
          });
  }

  /**
   * @brief Jacobian of LDG level `to` (flux and source) at all points of a batch of level to - 1: for a model with
   * AD jacobians the seed-stacked AD of ldg_evaluate_batch, otherwise its per-point jacobian_flux_source<to - 1,
   * to>.
   */
  template <uint to, typename Model, typename Batch, size_t n_out>
  void evaluate_ldg_level_jacobian(const Model &model,
                                   std::vector<PointJacobian<Batch::dim, n_out, Batch::n_fe_functions, 0>> &J,
                                   const Batch &batch, const size_t max_stacked,
                                   SeedStackWorkspace<Batch, n_out> &workspace)
  {
    internal::parallel_assign(J, batch.size(), {});
    if constexpr (internal::has_batch_ad_jacobians<Model>)
      internal::seed_stacked_jacobian<n_out, 1>(
          [&](auto &out, const auto &ad, size_t) { evaluate_ldg_level<to>(model, out, ad[0]); },
          [&](const size_t i, uint) -> auto & { return J[i]; }, std::array{&batch}, max_stacked,
          Term::flux | Term::source, workspace, internal::seeds_extractors<Model>);
    else
      internal::for_each_state(batch, [&](const size_t i, const auto &x, const auto &s) {
        model.template jacobian_flux_source<to - 1, to>(J[i].j_flux, J[i].j_source, x, s.values);
      });
  }

  /// Jacobian of the boundary numflux of LDG level `to`; see evaluate_ldg_level_jacobian.
  template <uint to, typename Model, typename Batch, size_t n_out>
  void
  evaluate_ldg_boundary_numflux_jacobian(const Model &model,
                                         std::vector<PointJacobian<Batch::dim, n_out, Batch::n_fe_functions, 0>> &J,
                                         const std::vector<dealii::Tensor<1, Batch::dim>> &normals, const Batch &batch,
                                         const size_t max_stacked, SeedStackWorkspace<Batch, n_out> &workspace)
  {
    internal::parallel_assign(J, batch.size(), {});
    if constexpr (internal::has_batch_ad_jacobians<Model>)
      internal::seed_stacked_jacobian<n_out, 1>(
          [&](auto &out, const auto &ad, const size_t n_blocks) {
            evaluate_ldg_boundary_numflux<to>(model, out, internal::stack_normals(workspace.normals, normals, n_blocks),
                                              ad[0]);
          },
          [&](const size_t i, uint) -> auto & { return J[i]; }, std::array{&batch}, max_stacked, Term::flux, workspace,
          internal::seeds_extractors<Model>);
    else
      internal::for_each_state(batch, [&](const size_t i, const auto &x, const auto &s) {
        model.template jacobian_boundary_numflux<to - 1, to>(J[i].j_flux, normals[i], x, s.values);
      });
  }

  /// Jacobian of the numerical flux of LDG level `to` with respect to both traces; see evaluate_ldg_level_jacobian.
  template <uint to, typename Model, typename Batch, size_t n_out>
  void evaluate_ldg_numflux_jacobian(const Model &model,
                                     std::vector<FaceJacobian<Batch::dim, n_out, Batch::n_fe_functions, 0>> &J,
                                     const std::vector<dealii::Tensor<1, Batch::dim>> &normals, const Batch &batch_s,
                                     const Batch &batch_n, const size_t max_stacked,
                                     SeedStackWorkspace<Batch, n_out, 2> &workspace)
  {
    using T1 = dealii::Tensor<1, Batch::dim>;
    internal::parallel_assign(J, batch_s.size(), {});
    if constexpr (internal::has_batch_ad_jacobians<Model>)
      internal::seed_stacked_jacobian<n_out, 2>(
          [&](auto &out, const auto &ad, const size_t n_blocks) {
            evaluate_ldg_numflux<to>(model, out, internal::stack_normals(workspace.normals, normals, n_blocks), ad[0],
                                     ad[1]);
          },
          [&](const size_t i, const uint side) -> auto & { return J[i][side]; }, std::array{&batch_s, &batch_n},
          max_stacked, Term::flux, workspace, internal::seeds_extractors<Model>);
    else
      internal::for_each_state_pair(batch_s, batch_n, [&](const size_t i, const auto &x, const auto &s, const auto &n) {
        std::array<SimpleMatrix<T1, n_out, Batch::n_fe_functions>, 2> jF;
        model.template jacobian_numflux<to - 1, to>(jF, normals[i], x, s.values, n.values);
        J[i][0].j_flux = jF[0];
        J[i][1].j_flux = jF[1];
      });
  }
} // namespace DiFfRG
