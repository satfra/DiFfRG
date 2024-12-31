#pragma once

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/csv_output.hh>
#include <DiFfRG/discretization/data/fe_output.hh>

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief Class to manage writing to files. FEM functions are written to vtk files and other data is written to csv
   * files.
   *
   * @tparam dim dimensionality of the FEM solution.
   * @tparam VectorType Vector type of the FEM solution.
   */
  template <uint dim, typename VectorType> class DataOutput
  {
  public:
    /**
     * @brief Construct a new Data Output object
     *
     * @param top_folder Folder where the output will be written, i.e. the folder containing the .pvd file and the .csv
     * files.
     * @param output_name Name of the output, i.e. the name of the .pvd file and the .csv files.
     * @param output_folder Folder where the .vtu files will be written. Should be relative to top_folder.
     * @param subdivisions Number of subdivisions of the cells in the .vtu files.
     */
    DataOutput(std::string top_folder, std::string output_name, std::string output_folder, const JSONValue &json);

    DataOutput(const JSONValue &json);

    /**
     * @brief Returns a reference to the FEOutput object used to write FEM functions to .vtu files and .pvd time series.
     */
    FEOutput<dim, VectorType> &fe_output();

    /**
     * @brief Returns a reference to the CsvOutput object associated with the given name, which is used to write scalar
     * values to .csv files. If the object does not exist, it is created.
     */
    CsvOutput &csv_file(const std::string &name);

    /**
     * @brief Save all attached data vectors to a .vtu and append it to the time series. Also flush any attached scalar
     * values to the .csv files.
     *
     * @param time tag attached in the .pvd time series and the .csv files.
     */
    void flush(const double time);

    /**
     * @brief Set the value of Lambda. If Lambda is set, the output will contain a column for k = exp(-t) * Lambda.
     *
     * @param Lambda The value of Lambda.
     */
    void set_Lambda(const double Lambda);

    /**
     * @brief Get the name of the output.
     *
     * @return const std::string&
     */
    const std::string &get_output_name() const;

    /**
     * @brief Dump a vector of vectors to a .csv file, e.g. for a higher-dimensional grid function.
     *
     * @param name Name of the .csv file.
     * @param values Vector of vectors to dump.
     * @param attach If true, the values are appended to the file. If false, the file is truncated.
     * @param header Header of the .csv file. If attach is true, this is ignored.
     */
    void dump_to_csv(const std::string &name, const std::vector<std::vector<double>> &values, bool attach = false,
                     const std::vector<std::string> header = {});

  private:
    JSONValue json;
    const std::string top_folder;
    const std::string output_name;
    const std::string output_folder;
    double Lambda;

    std::vector<double> time_values;
    std::vector<double> k_values;

    FEOutput<dim, VectorType> fe_out;
    std::map<std::string, CsvOutput> csv_files;
  };

  template <> class DataOutput<0, Vector<double>>
  {
    static constexpr uint dim = 0;
    using VectorType = Vector<double>;

  public:
    /**
     * @brief Construct a new Data Output object, not associated with a DoFHandler.
     *
     * @param top_folder Folder where the output will be written, i.e. the folder containing the .pvd file and the .csv
     * files.
     * @param output_name Name of the output, i.e. the name of the .pvd file and the .csv files.
     * @param output_folder Folder where the .vtu files will be written. Should be relative to top_folder.
     * @param subdivisions Number of subdivisions of the cells in the .vtu files.
     */
    DataOutput(std::string top_folder, std::string output_name, std::string output_folder, const JSONValue &json);

    DataOutput(const JSONValue &json);

    /**
     * @brief Returns a reference to the FEOutput object used to write FEM functions to .vtu files and .pvd time series.
     * In this case (d=0), the object is empty.
     */
    FEOutput<dim, VectorType> &fe_output();

    /**
     * @brief Returns a reference to the CsvOutput object associated with the given name, which is used to write scalar
     * values to .csv files. If the object does not exist, it is created.
     */
    CsvOutput &csv_file(const std::string &name);

    /**
     * @brief Save all attached data vectors to a .vtu and append it to the time series. Also flush any attached scalar
     * values to the .csv files.
     *
     * @param time tag attached in the .pvd time series and the .csv files.
     */
    void flush(const double time);

    /**
     * @brief Set the value of Lambda. If Lambda is set, the output will contain a column for k = exp(-t) * Lambda.
     *
     * @param Lambda The value of Lambda.
     */
    void set_Lambda(const double Lambda);

    /**
     * @brief Get the name of the output.
     *
     * @return const std::string&
     */
    const std::string &get_output_name() const;

    /**
     * @brief Dump a vector of vectors to a .csv file, e.g. for a higher-dimensional grid function.
     *
     * @param name Name of the .csv file.
     * @param values Vector of vectors to dump.
     * @param attach If true, the values are appended to the file. If false, the file is truncated.
     * @param header Header of the .csv file. If attach is true, this is ignored.
     */
    void dump_to_csv(const std::string &name, const std::vector<std::vector<double>> &values, bool attach = false,
                     const std::vector<std::string> header = {});

  private:
    JSONValue json;
    const std::string top_folder;
    const std::string output_name;
    const std::string output_folder;
    double Lambda;

    std::vector<double> time_values;
    std::vector<double> k_values;

    FEOutput<dim, VectorType> fe_out;
    std::map<std::string, CsvOutput> csv_files;
  };
} // namespace DiFfRG