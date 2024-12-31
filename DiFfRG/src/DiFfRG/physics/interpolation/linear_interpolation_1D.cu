// DiFfRG
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation/linear_interpolation_1D.hh>

namespace DiFfRG
{
  template <typename NT, typename Coordinates>
  LinearInterpolator1D<NT, Coordinates>::LinearInterpolator1D(const std::vector<NT> &data,
                                                              const Coordinates &coordinates)
      : LinearInterpolator1D(data.data(), coordinates)
  {
    if (data.size() != coordinates.size()) throw std::runtime_error("Data and coordinates must have the same size");
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator1D<NT, Coordinates>::LinearInterpolator1D(const Coordinates &coordinates)
      : LinearInterpolator1D(std::vector<NT>(coordinates.size(), NT(0)), coordinates)
  {
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator1D<NT, Coordinates>::LinearInterpolator1D(const NT *in_data, const Coordinates &coordinates)
      : size(coordinates.size()), coordinates(coordinates), m_data(nullptr), owner(true)
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
    check_cuda("LinearInterpolator1D::LinearInterpolator1D");
    device_data_ptr = thrust::raw_pointer_cast(device_data->data());
#endif
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator1D<NT, Coordinates>::LinearInterpolator1D(const LinearInterpolator1D<NT, Coordinates> &other)
      : size(other.size), coordinates(other.coordinates), owner(false)
  {
    m_data = other.m_data;
#ifdef __CUDACC__
    device_data_ptr = other.device_data_ptr;
#endif
  }

  template <typename NT, typename Coordinates> void LinearInterpolator1D<NT, Coordinates>::update()
  {
    if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

#ifdef __CUDACC__
    thrust::copy(m_data.get(), m_data.get() + size, device_data->begin());
    check_cuda("LinearInterpolator1D::update");
#endif
  }

  template <typename NT, typename Coordinates> NT *LinearInterpolator1D<NT, Coordinates>::data() const
  {
    return m_data.get();
  }

  template <typename NT, typename Coordinates> NT &LinearInterpolator1D<NT, Coordinates>::operator[](const uint i)
  {
    return m_data[i];
  }
  template <typename NT, typename Coordinates>
  const NT &LinearInterpolator1D<NT, Coordinates>::operator[](const uint i) const
  {
    return m_data[i];
  }

  template class LinearInterpolator1D<double, LinearCoordinates1D<float>>;
  template class LinearInterpolator1D<float, LinearCoordinates1D<float>>;
  template class LinearInterpolator1D<autodiff::real, LinearCoordinates1D<float>>;
  template class LinearInterpolator1D<double, LogarithmicCoordinates1D<float>>;
  template class LinearInterpolator1D<float, LogarithmicCoordinates1D<float>>;
  template class LinearInterpolator1D<autodiff::real, LogarithmicCoordinates1D<float>>;

  template class LinearInterpolator1D<double, LinearCoordinates1D<double>>;
  template class LinearInterpolator1D<float, LinearCoordinates1D<double>>;
  template class LinearInterpolator1D<autodiff::real, LinearCoordinates1D<double>>;
  template class LinearInterpolator1D<double, LogarithmicCoordinates1D<double>>;
  template class LinearInterpolator1D<float, LogarithmicCoordinates1D<double>>;
  template class LinearInterpolator1D<autodiff::real, LogarithmicCoordinates1D<double>>;
} // namespace DiFfRG
