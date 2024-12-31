#pragma once

// DiFfRG
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/discretization/mesh/no_adaptivity.hh>

namespace DiFfRG
{
  /**
   * @brief The abstract base class for all timestepping algorithms.
   * It provides a standard constructor which populates typical timestepping parameters from a given JSONValue object,
   * such as the timestep sizes, tolerances, verbosity, etc. that are used in the timestepping algorithms.
   *
   * In the JSONValue object, a /timestepping/ section must be present with the following parameters:
   * - /timestepping/output_dt: The output timestep size.
   * - /timestepping/implicit/dt: The timestep size for an implicit timestepping algorithm.
   * - /timestepping/implicit/minimal_dt: The minimal timestep size for an implicit timestepping algorithm.
   * - /timestepping/implicit/maximal_dt: The maximal timestep size for an implicit timestepping algorithm.
   * - /timestepping/implicit/abs_tol: The absolute tolerance for an implicit timestepping algorithm.
   * - /timestepping/implicit/rel_tol: The relative tolerance for an implicit timestepping algorithm.
   * - /timestepping/explicit/dt: The timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/minimal_dt: The minimal timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/maximal_dt: The maximal timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/abs_tol: The absolute tolerance for an explicit timestepping algorithm.
   * - /timestepping/explicit/rel_tol: The relative tolerance for an explicit timestepping algorithm.
   *
   * Additionally, the following parameters are being used:
   * - /output/verbosity: The verbosity level of the output.
   * - /physical/Lambda: The RG scale parameter Lambda. If not present, no RG scale is given when console_out is called.
   *
   * The console_out method is used to print information about the current time, the calculation time and the current
   * RG scale k to the console in a standardized way.
   *
   * @tparam VectorType_ The type of the vector used in the timestepping algorithm.. Currently only Vector<double> is
   * supported.
   * @tparam SparseMatrixType_ The type of the sparse matrix used in the timestepping algorithm. This depends on the
   * assembler used in the computation.
   * @tparam dim_ The dimensionality of the spatial discretization.
   */
  template <typename VectorType_, typename SparseMatrixType_, uint dim_> class AbstractTimestepper
  {
  protected:
    static constexpr uint dim = dim_;
    using VectorType = VectorType_;
    using NumberType = typename get_type::NumberType<VectorType>;
    using SparseMatrixType = SparseMatrixType_;
    using InverseSparseMatrixType = typename get_type::InverseSparseMatrixType<SparseMatrixType>;
    static_assert(std::is_same_v<VectorType, Vector<NumberType>>, "VectorType must not be Vector<double>!");
    using BlockVectorType = dealii::BlockVector<NumberType>;

  public:
    /**
     * @brief Construct a new Abstract Timestepper object
     *
     * @param json The JSONValue object must contain a /timestepping/ section with all necessary parameters.
     * @param assembler
     * @param data_out
     * @param adaptor
     */
    AbstractTimestepper(const JSONValue &json, AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler,
                        DataOutput<dim, VectorType> *data_out = nullptr, AbstractAdaptor<VectorType> *adaptor = nullptr)
        : json(json), assembler(assembler), data_out(data_out), adaptor(adaptor),
          start_time(std::chrono::high_resolution_clock::now())
    {
      verbosity = json.get_int("/output/verbosity");
      output_dt = json.get_double("/timestepping/output_dt");

      impl.dt = json.get_double("/timestepping/implicit/dt");
      impl.minimal_dt = json.get_double("/timestepping/implicit/minimal_dt");
      impl.maximal_dt = json.get_double("/timestepping/implicit/maximal_dt");
      impl.abs_tol = json.get_double("/timestepping/implicit/abs_tol");
      impl.rel_tol = json.get_double("/timestepping/implicit/rel_tol");

      expl.dt = json.get_double("/timestepping/explicit/dt");
      expl.minimal_dt = json.get_double("/timestepping/explicit/minimal_dt");
      expl.maximal_dt = json.get_double("/timestepping/explicit/maximal_dt");
      expl.abs_tol = json.get_double("/timestepping/explicit/abs_tol");
      expl.rel_tol = json.get_double("/timestepping/explicit/rel_tol");

      try {
        Lambda = json.get_double("/physical/Lambda");
      } catch (std::exception &e) {
        Lambda = -1.0;
      }
    }

    /**
     * @brief Utility function to obtain a DataOutput object. If no DataOutput object is provided, a default one is
     * created.
     *
     * @return DataOutput<dim, VectorType>* A pointer to the DataOutput object.
     */
    DataOutput<dim, VectorType> *get_data_out()
    {
      if (data_out == nullptr) {
        data_out_default = std::make_shared<DataOutput<dim, VectorType>>(json);
        return data_out_default.get();
      }
      return data_out;
    }

    /**
     * @brief Utility function to obtain an Adaptor object. If no Adaptor object is provided, a default one is created,
     * which is the NoAdaptivity object, i.e. no mesh adaptivity is used.
     *
     * @return AbstractAdaptor<VectorType>* A pointer to the Adaptor object.
     */
    AbstractAdaptor<VectorType> *get_adaptor()
    {
      if (adaptor == nullptr) {
        adaptor_default = std::make_shared<NoAdaptivity<VectorType>>();
        return adaptor_default.get();
      }
      return adaptor;
    }

    /**
     * @brief Any derived class must implement this method to run the timestepping algorithm.
     *
     * @param initial_condition A pointer to a flowing variables object that contains the initial condition.
     * @param t_start The start time of the simulation.
     * @param t_stop The run method will evolve the system from t_start to t_stop.
     */
    virtual void run(AbstractFlowingVariables<NumberType> *initial_condition, const double t_start,
                     const double t_stop) = 0;

  protected:
    const JSONValue json;
    AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler;
    DataOutput<dim, VectorType> *data_out;
    AbstractAdaptor<VectorType> *adaptor;

    const std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

    std::shared_ptr<NoAdaptivity<VectorType>> adaptor_default;
    std::shared_ptr<DataOutput<dim, VectorType>> data_out_default;

    double Lambda;
    int verbosity;
    double output_dt;
    struct implicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
    } impl;

    struct explicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
    } expl;

    /**
     * @brief Pretty-print the status of the timestepping algorithm to the console.
     *
     * @param t Current time.
     * @param name A tag prepended to the output.
     * @param verbosity_level The verbosity level of the output.
     * @param calc_dt_ms If >= 0, the time in milliseconds it took to calculate the current timestep.
     */
    void console_out(const double t, const std::string name, const int verbosity_level,
                     const double calc_dt_ms = -1.0) const
    {
      if (verbosity >= verbosity_level) {
        std::ios_base::fmtflags oldflags = std::cout.flags();
        std::cout << "[" << name << "]";
        std::cout << "      t: " << std::setw(10) << std::left << std::setprecision(4) << std::scientific << t;
        if (Lambda > 0.0) {
          std::cout << " | k: " << std::setw(10) << std::left << std::setprecision(4) << std::scientific
                    << exp(-t) * Lambda;
        }
        if (calc_dt_ms >= 0.0) {
          std::cout << " | calc_dt: " << time_format_ms(size_t(calc_dt_ms));
        }
        size_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
        std::cout << " | calc_t: " << time_format_ms(milliseconds);
        std::cout << std::endl;
        std::cout.flags(oldflags);
      }
    }
  };
} // namespace DiFfRG