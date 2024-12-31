// DiFfRG
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation/linear_interpolation_3D.hh>

namespace DiFfRG
{
  template <typename NT, typename Coordinates>
  LinearInterpolator3D<NT, Coordinates>::LinearInterpolator3D(const std::vector<NT> &data,
                                                              const Coordinates &coordinates)
      : LinearInterpolator3D(data.data(), coordinates)
  {
    if (data.size() != coordinates.size()) throw std::runtime_error("Data and coordinates must have the same size");
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator3D<NT, Coordinates>::LinearInterpolator3D(const Coordinates &coordinates)
      : LinearInterpolator3D(std::vector<NT>(coordinates.size(), NT(0)), coordinates)
  {
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator3D<NT, Coordinates>::LinearInterpolator3D(const NT *in_data, const Coordinates &coordinates)
      : size(coordinates.size()), shape(coordinates.sizes()), coordinates(coordinates), m_data(nullptr), owner(true)
  {
    // Copy input data
    m_data = std::shared_ptr<NT[]>(new NT[size]);
    for (uint i = 0; i < size; ++i)
      m_data[i] = NT(in_data[i]);
      // Create the thrust::device_vector
#ifdef __CUDACC__
    device_data = std::shared_ptr<thrust::device_vector<NT>>(new thrust::device_vector<NT>());
    device_data->resize(size);
    thrust::copy(m_data.get(), m_data.get() + size, device_data->begin());
    check_cuda("LinearInterpolator3D::LinearInterpolator3D");
    device_data_ptr = thrust::raw_pointer_cast(device_data->data());
#endif
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator3D<NT, Coordinates>::LinearInterpolator3D(const LinearInterpolator3D<NT, Coordinates> &other)
      : size(other.size), coordinates(other.coordinates), owner(false), shape(other.shape)
  {
    m_data = other.m_data;
#ifdef __CUDACC__
    device_data_ptr = other.device_data_ptr;
#endif
  }

  template <typename NT, typename Coordinates> void LinearInterpolator3D<NT, Coordinates>::update()
  {
    if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

#ifdef __CUDACC__
    thrust::copy(m_data.get(), m_data.get() + size, device_data->begin());
    check_cuda("LinearInterpolator3D::update");
#endif
  }

  template <typename NT, typename Coordinates> NT *LinearInterpolator3D<NT, Coordinates>::data() const
  {
    return m_data.get();
  }

  template <typename NT, typename Coordinates> NT &LinearInterpolator3D<NT, Coordinates>::operator[](const uint i)
  {
    return m_data[i];
  }
  template <typename NT, typename Coordinates>
  const NT &LinearInterpolator3D<NT, Coordinates>::operator[](const uint i) const
  {
    return m_data[i];
  }

  template class LinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>,
                                                               LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      float, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class LinearInterpolator3D<
      double,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      float,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class LinearInterpolator3D<
      double, CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      float, CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class LinearInterpolator3D<
      double,
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<
      float,
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;

  template class LinearInterpolator3D<double,
                                      CoordinatePackND<LogarithmicCoordinates1D<double>,
                                                       LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<float,
                                      CoordinatePackND<LogarithmicCoordinates1D<double>,
                                                       LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<autodiff::real,
                                      CoordinatePackND<LogarithmicCoordinates1D<double>,
                                                       LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>>;

  template class LinearInterpolator3D<
      double, CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<
      float, CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;

} // namespace DiFfRG
