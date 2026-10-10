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
   * @brief Interpolates columns of tabulated data read from .csv files, as functions of one chosen column \f$x\f$.
   *
   * The columns of all files are placed side by side and numbered consecutively: with two files of three columns
   * each, the columns of the second file are 3, 4, 5. All files must have the same number of rows. Rows that contain
   * a NaN in any column are dropped. Column `i` is then interpolated as a function of column `x_column` with a
   * barycentric rational interpolant (see Interpolation::Barycentric).
   *
   * Outside the range of the data, value() returns the value at the nearest end of the range and derivative() the
   * derivative there. The constructor checks that the interpolant reproduces every data point to a relative
   * precision of \f$10^{-4}\f$ and throws otherwise.
   *
   * @code
   * // data.csv: columns p, Z(p), m2(p)
   * ExternalDataInterpolator data({"data.csv"});
   * const double Z = data.value(p, 1);
   * @endcode
   */
  class ExternalDataInterpolator
  {
  public:
    /**
     * @brief Read and interpolate the data.
     *
     * @param input_files the .csv file(s) to read.
     * @param post_processors functions applied to every entry of a column after reading, `post_processors[c]` to
     * column `c` (counted as above, including the \f$x\f$ column); may be shorter than the number of columns.
     * @param separator the field separator.
     * @param has_header whether the first row of each file is a header.
     * @param x_column the column holding \f$x\f$.
     * @param order the approximation order of the barycentric interpolant, see Interpolation::Barycentric.
     */
    ExternalDataInterpolator(std::vector<std::string> input_files,
                             std::vector<std::function<double(double)>> post_processors, char separator = ',',
                             bool has_header = false, uint x_column = 0, uint order = 1);

    /// As above, without post-processing.
    ExternalDataInterpolator(std::vector<std::string> input_files, char separator = ',', bool has_header = false,
                             uint x_column = 0, uint order = 1);

    /// An empty interpolator; must be assigned before it is evaluated.
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
    /// Column `i` at \f$x\f$; clamped to the nearest end of the data range.
    double value(double x, uint i) const;

    /// Derivative of column `i` with respect to \f$x\f$; outside the data range, the derivative at the nearest end.
    double derivative(double x, uint i) const;

  private:
    std::string input_file;
    std::vector<std::vector<double>> file_data;
    std::vector<boost::math::interpolators::barycentric_rational<double>> interpolators;

    double max_x, min_x;
  };
} // namespace DiFfRG
