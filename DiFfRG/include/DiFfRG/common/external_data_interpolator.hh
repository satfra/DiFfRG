#pragma once

// standard library
#include <functional>
#include <string>
#include <vector>

// external libraries
#include <boost/math/interpolators/barycentric_rational.hpp>

namespace DiFfRG
{
  /**
   * @brief This class takes in a .csv file with x-dependent data and interpolates it to a given x on request.
   *
   * Data can be read from several files which are identically formatted and will then be concatenated. The choice which
   * column represents the independent x variable can be made by the user when creating the object. After read-in, the
   * data can be post-processed before interpolation. After the object is constructed, this class allows to access both
   * the interpolant and its derivative.
   *
   * Note, that interpolations beyond the range of the data will return the value at the closest boundary.
   *
   * Internally, the boost barycentric rational interpolator is used.
   *
   */
  class ExternalDataInterpolator
  {
  public:
    /**
     * @brief Construct a new External Data Interpolator object
     *
     * @param input_files The .csv file(s) to read the data from.
     * @param post_processors The post processors to apply to the data.
     * @param separator The separator used in the .csv file.
     * @param has_header Whether the .csv file has a header.
     * @param x_column The column in the .csv file(s) that contains the x values. This counts all columns, including the
     * ones from previous files.
     */
    ExternalDataInterpolator(std::vector<std::string> input_files,
                             std::vector<std::function<double(double)>> post_processors, char separator = ',',
                             bool has_header = false, uint x_column = 0, uint order = 1);

    /**
     * @brief Construct a new External Data Interpolator object
     *
     * @param input_files The .csv file(s) to read the data from.
     * @param separator The separator used in the .csv file.
     * @param has_header Whether the .csv file has a header.
     * @param x_column The column in the .csv file(s) that contains the x values. This counts all columns, including the
     * ones from previous files.
     */
    ExternalDataInterpolator(std::vector<std::string> input_files, char separator = ',', bool has_header = false,
                             uint x_column = 0, uint order = 1);

    /**
     * @brief Construct an empty External Data Interpolator object
     *
     */
    ExternalDataInterpolator();

  private:
    /**
     * @brief Set up the interpolator.
     *
     * @param input_files The .csv file(s) to read the data from.
     * @param post_processors The post processors to apply to the data.
     * @param separator The separator used in the .csv file.
     * @param has_header Whether the .csv file has a header.
     * @param x_column The column in the .csv file(s) that contains the x values. This counts all columns, including the
     * ones from previous files.
     */
    void setup(const std::vector<std::string> &input_files,
               const std::vector<std::function<double(double)>> &post_processors, char separator = ',',
               bool has_header = false, uint x_column = 0, uint order = 1);

    /**
     * @brief Check whether the interpolator is consistent with the original data.
     * The interpolator checks if it can reproduce the original data with a given relative precision.
     *
     * @param tolerance The relative precision to use.
     * @param x_column The column holding the independent variable, which the interpolants are
     * evaluated at.
     * @return true If the interpolator is consistent with the original data.
     */
    bool check_consistency(double tolerance, uint x_column) const;

  public:
    /**
     * @brief Interpolate the data to a given x.
     *
     * @param x The x to interpolate to.
     * @param i The index of the data to interpolate.
     * @return double The interpolated data.
     */
    double value(double x, uint i) const;

    /**
     * @brief Interpolate the derivative of the data to a given x.
     *
     * @param x The x to interpolate to.
     * @param i The index of the data to interpolate.
     * @return double The interpolated derivative.
     */
    double derivative(double x, uint i) const;

  private:
    std::string input_file;
    std::vector<std::vector<double>> file_data;
    std::vector<boost::math::interpolators::barycentric_rational<double>> interpolators;

    double max_x, min_x;
  };
} // namespace DiFfRG
