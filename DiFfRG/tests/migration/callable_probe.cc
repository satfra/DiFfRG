#include <DiFfRG/DiFfRG.hh>

void probe(const DiFfRG::ConfigTree &json)
{
  DiFfRG::ConfigurationHelper config(json);
  (void)config.get_log_file();
  (void)config.get_output_name();
  (void)config.get_output_folder();
  (void)config.get_top_folder();
  DiFfRG::CsvOutput csv(".", "run", json);
  DiFfRG::FEOutput<1, dealii::Vector<double>> fe(".", "run", "fields", json);
  DiFfRG::HDF5Output hdf5(".", "run.h5", json);
  DiFfRG::RectangularMesh<1> mesh(json);
}
