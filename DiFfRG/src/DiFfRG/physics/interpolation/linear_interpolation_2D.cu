// DiFfRG
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation/linear_interpolation_2D.hh>

namespace DiFfRG
{
  template <typename NT, typename Coordinates>
  LinearInterpolator2D<NT, Coordinates>::LinearInterpolator2D(const std::vector<NT> &data,
                                                              const Coordinates &coordinates)
      : LinearInterpolator2D(data.data(), coordinates)
  {
    if (data.size() != coordinates.size()) throw std::runtime_error("Data and coordinates must have the same size");
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator2D<NT, Coordinates>::LinearInterpolator2D(const Coordinates &coordinates)
      : LinearInterpolator2D(std::vector<NT>(coordinates.size(), NT(0)), coordinates)
  {
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator2D<NT, Coordinates>::LinearInterpolator2D(const NT *in_data, const Coordinates &coordinates)
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
    check_cuda("LinearInterpolator2D::LinearInterpolator2D");
    device_data_ptr = thrust::raw_pointer_cast(device_data->data());
#endif
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator2D<NT, Coordinates>::LinearInterpolator2D(const LinearInterpolator2D<NT, Coordinates> &other)
      : size(other.size), coordinates(other.coordinates), owner(false), shape(other.shape)
  {
    m_data = other.m_data;
#ifdef __CUDACC__
    device_data_ptr = other.device_data_ptr;
#endif
  }

  template <typename NT, typename Coordinates> void LinearInterpolator2D<NT, Coordinates>::update()
  {
    if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

#ifdef __CUDACC__
    thrust::copy(m_data.get(), m_data.get() + size, device_data->begin());
    check_cuda("LinearInterpolator2D::update");
#endif
  }

  template <typename NT, typename Coordinates> NT *LinearInterpolator2D<NT, Coordinates>::data() const
  {
    return m_data.get();
  }

  template <typename NT, typename Coordinates> NT &LinearInterpolator2D<NT, Coordinates>::operator[](const uint i)
  {
    return m_data[i];
  }
  template <typename NT, typename Coordinates>
  const NT &LinearInterpolator2D<NT, Coordinates>::operator[](const uint i) const
  {
    return m_data[i];
  }

  template class LinearInterpolator2D<
      double, CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>>>;
  template class LinearInterpolator2D<
      float, CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>>>;
  template class LinearInterpolator2D<
      autodiff::real, CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>>>;

  template class LinearInterpolator2D<double, CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator2D<float, CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator2D<autodiff::real,
                                      CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class LinearInterpolator2D<
      double, CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>>>;
  template class LinearInterpolator2D<
      float, CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>>>;
  template class LinearInterpolator2D<
      autodiff::real, CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>>>;

  template class LinearInterpolator2D<double,
                                      CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator2D<float,
                                      CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator2D<autodiff::real,
                                      CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;

} // namespace DiFfRG
