// standard library
#include <fstream>

// external libraries
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/data_output.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <uint dim, typename VectorType>
  DataOutput<dim, VectorType>::DataOutput(std::string top_folder, std::string output_name, std::string output_folder,
                                          const JSONValue &json)
      : json(json), top_folder(make_folder(top_folder)), output_name(output_name),
        output_folder(make_folder(output_folder)), Lambda(-1.), fe_out(top_folder, output_name, output_folder, json)
  {
    set_Lambda(json.get_double("/physical/Lambda"));
  }

  template <uint dim, typename VectorType>
  DataOutput<dim, VectorType>::DataOutput(const JSONValue &json)
      : DataOutput(json.get_string("/output/folder"), json.get_string("/output/name"), json.get_string("/output/name"),
                   json)
  {
  }

  template <uint dim, typename VectorType> FEOutput<dim, VectorType> &DataOutput<dim, VectorType>::fe_output()
  {
    return fe_out;
  }

  template <uint dim, typename VectorType> CsvOutput &DataOutput<dim, VectorType>::csv_file(const std::string &name)
  {
    const auto obj = csv_files.find(name);
    if (obj == csv_files.end()) {
      std::string filename = output_name + "_" + name;
      if (name.find("/") != std::string::npos) filename = name;
      auto ret = csv_files.emplace(name, CsvOutput(top_folder, filename, json));
      return ret.first->second;
    }
    return obj->second;
  }

  template <uint dim, typename VectorType> const std::string &DataOutput<dim, VectorType>::get_output_name() const
  {
    return output_name;
  }

  template <uint dim, typename VectorType> void DataOutput<dim, VectorType>::flush(const double time)
  {
    time_values.push_back(time);
    k_values.push_back(std::exp(-time) * Lambda);

    fe_out.flush(time);
    for (auto &csv : csv_files)
      csv.second.flush(time);
  }

  template <uint dim, typename VectorType>
  void DataOutput<dim, VectorType>::dump_to_csv(const std::string &name, const std::vector<std::vector<double>> &values,
                                                bool attach, const std::vector<std::string> header)
  {
    std::ofstream output_stream(top_folder + name, attach ? std::ofstream::app : std::ofstream::trunc);
    output_stream << std::scientific;
    if (!attach)
      if (header.size() > 0) {
        for (uint i = 0; i < header.size(); ++i) {
          const auto &value = header[i];
          output_stream << strip_name(value);
          if (i != header.size() - 1) output_stream << ",";
        }
        output_stream << std::endl;
      }

    for (uint i = 0; i < values.size(); ++i) {
      const auto &row = values[i];
      for (uint i = 0; i < row.size(); ++i) {
        output_stream << row[i];
        if (i != row.size() - 1) output_stream << ",";
      }
      output_stream << std::endl;
    }
  }

  template <uint dim, typename VectorType> void DataOutput<dim, VectorType>::set_Lambda(const double Lambda)
  {
    // This needs safety checks, so that Lambda can only be set once, before any call to flush().
    if (time_values.size() > 0 && !is_close(this->Lambda, Lambda))
      throw std::runtime_error("Lambda has either already been set or there has been an attempt to change it.");
    this->Lambda = Lambda;

    for (auto &csv : csv_files)
      csv.second.set_Lambda(Lambda);
  }

  template class DataOutput<1, dealii::Vector<double>>;
  template class DataOutput<2, dealii::Vector<double>>;
  template class DataOutput<3, dealii::Vector<double>>;
  template class DataOutput<1, dealii::BlockVector<double>>;
  template class DataOutput<2, dealii::BlockVector<double>>;
  template class DataOutput<3, dealii::BlockVector<double>>;

  DataOutput<0, Vector<double>>::DataOutput(std::string top_folder, std::string output_name, std::string output_folder,
                                            const JSONValue &json)
      : json(json), top_folder(make_folder(top_folder)), output_name(output_name),
        output_folder(make_folder(output_folder)), Lambda(-1.)
  {
  }

  DataOutput<0, Vector<double>>::DataOutput(const JSONValue &json)
      : DataOutput(json.get_string("/output/folder"), json.get_string("/output/name"), json.get_string("/output/name"),
                   json)
  {
  }

  FEOutput<0, Vector<double>> &DataOutput<0, Vector<double>>::fe_output() { return fe_out; }

  CsvOutput &DataOutput<0, Vector<double>>::csv_file(const std::string &name)
  {
    const auto obj = this->csv_files.find(name);
    if (obj == csv_files.end()) {
      std::string filename = output_name + "_" + name;
      if (name.find("/") != std::string::npos) filename = name;
      auto ret = csv_files.emplace(name, CsvOutput(top_folder, filename, json));
      return ret.first->second;
    }
    return obj->second;
  }

  const std::string &DataOutput<0, Vector<double>>::get_output_name() const { return output_name; }

  void DataOutput<0, Vector<double>>::flush(double time)
  {
    time_values.push_back(time);
    k_values.push_back(std::exp(-time) * Lambda);

    fe_out.flush(time);
    for (auto &csv : csv_files)
      csv.second.flush(time);
  }

  void DataOutput<0, Vector<double>>::dump_to_csv(const std::string &name,
                                                  const std::vector<std::vector<double>> &values, bool attach,
                                                  const std::vector<std::string> header)
  {
    std::ofstream output_stream(top_folder + name, attach ? std::ofstream::app : std::ofstream::trunc);
    output_stream << std::scientific;
    if (!attach)
      if (header.size() > 0) {
        for (uint i = 0; i < header.size(); ++i) {
          const auto &value = header[i];
          output_stream << strip_name(value);
          if (i != header.size() - 1) output_stream << ",";
        }
        output_stream << std::endl;
      }

    for (uint i = 0; i < values.size(); ++i) {
      const auto &row = values[i];
      for (uint i = 0; i < row.size(); ++i) {
        output_stream << row[i];
        if (i != row.size() - 1) output_stream << ",";
      }
      output_stream << std::endl;
    }
  }

  void DataOutput<0, Vector<double>>::set_Lambda(const double Lambda)
  {
    // This needs safety checks, so that Lambda can only be set once, before any call to flush().
    if (time_values.size() > 0 && !is_close(this->Lambda, Lambda))
      throw std::runtime_error("Lambda has either already been set or there has been an attempt to change it.");
    this->Lambda = Lambda;

    for (auto &csv : csv_files)
      csv.second.set_Lambda(Lambda);
  }
} // namespace DiFfRG