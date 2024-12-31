// DiFfRG
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation/tex_linear_interpolation_1D.hh>

namespace DiFfRG
{
  template <typename NT, typename Coordinates>
  TexLinearInterpolator1D<NT, Coordinates>::TexLinearInterpolator1D(const std::vector<NT> &data,
                                                                    const Coordinates &coordinates)
      : TexLinearInterpolator1D(data.data(), coordinates)
  {
    if (data.size() != coordinates.size()) throw std::runtime_error("Data and coordinates must have the same size");
  }

  template <typename NT, typename Coordinates>
  TexLinearInterpolator1D<NT, Coordinates>::TexLinearInterpolator1D(const Coordinates &coordinates)
      : TexLinearInterpolator1D(std::vector<NT>(coordinates.size(), NT(0)), coordinates)
  {
  }

  template <typename NT, typename Coordinates>
  TexLinearInterpolator1D<NT, Coordinates>::TexLinearInterpolator1D(const NT *in_data, const Coordinates &coordinates)
      : size(coordinates.size()), coordinates(coordinates), m_data(nullptr), m_data_AD(nullptr), owner(true),
        n_devices(0)
  {
#ifdef __CUDACC__
    cudaGetDeviceCount(&n_devices);
    if (n_devices == 0) throw std::runtime_error("No CUDA devices found");
    if (n_devices > max_device_count)
      throw std::runtime_error("Too many CUDA devices found - TexLinearInterpolator1D supports at most " +
                               std::to_string(max_device_count) + " devices");
#endif

    if constexpr (std::is_same_v<float, ReturnType>) {
      // Copy input data and possibly cast to float
      m_data = std::shared_ptr<float[]>(new float[size]);
      for (uint i = 0; i < size; ++i)
        m_data[i] = static_cast<float>(in_data[i]);

#ifdef __CUDACC__
      device_array.resize(n_devices, nullptr);

      for (int device = 0; device < n_devices; ++device) {
        cudaSetDevice(device);

        // Allocate device array and copy data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&device_array[device], &channelDesc, size);
        cudaMemcpy2DToArray(device_array[device], 0, 0, &m_data[0], sizeof(m_data[0]) * size, sizeof(m_data[0]) * size,
                            1, cudaMemcpyHostToDevice);

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = device_array[device];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaCreateTextureObject(&texture[device], &resDesc, &texDesc, NULL);
        cudaDeviceSynchronize();
        check_cuda("TexLinearInterpolator1D::TexLinearInterpolator1D");
      }
      cudaSetDevice(0);
#endif
    } else if constexpr (std::is_same_v<autodiff::real, ReturnType>) {
      // Copy input data and possibly cast to float
      m_data = std::shared_ptr<float[]>(new float[size]);
      m_data_AD = std::shared_ptr<float[]>(new float[size]);
      for (uint i = 0; i < size; ++i) {
        m_data[i] = static_cast<float>(val(in_data[i]));
        m_data_AD[i] = static_cast<float>(derivative(in_data[i]));
      }

#ifdef __CUDACC__
      device_array.resize(n_devices, nullptr);
      device_array_AD.resize(n_devices, nullptr);

      for (int device = 0; device < n_devices; ++device) {
        cudaSetDevice(device);

        // Allocate device array and copy data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&device_array[device], &channelDesc, size);
        cudaMallocArray(&device_array_AD[device], &channelDesc, size);
        cudaMemcpy2DToArray(device_array[device], 0, 0, &m_data[0], sizeof(m_data[0]) * size, sizeof(m_data[0]) * size,
                            1, cudaMemcpyHostToDevice);
        cudaMemcpy2DToArray(device_array_AD[device], 0, 0, &m_data_AD[0], sizeof(m_data_AD[0]) * size,
                            sizeof(m_data_AD[0]) * size, 1, cudaMemcpyHostToDevice);

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = device_array[device];

        struct cudaResourceDesc resDesc_AD;
        memset(&resDesc_AD, 0, sizeof(resDesc_AD));
        resDesc_AD.resType = cudaResourceTypeArray;
        resDesc_AD.res.array.array = device_array_AD[device];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaCreateTextureObject(&texture[device], &resDesc, &texDesc, NULL);
        cudaCreateTextureObject(&texture_AD[device], &resDesc_AD, &texDesc, NULL);
        cudaDeviceSynchronize();
        check_cuda("TexLinearInterpolator1D::TexLinearInterpolator1D");
      }
      cudaSetDevice(0);
#endif
    }
  }

  template <typename NT, typename Coordinates>
  TexLinearInterpolator1D<NT, Coordinates>::TexLinearInterpolator1D(
      const TexLinearInterpolator1D<NT, Coordinates> &other)
      : size(other.size), coordinates(other.coordinates), owner(false), n_devices(other.n_devices)
  {
    m_data = other.m_data;
    m_data_AD = other.m_data_AD;
#ifdef __CUDACC__
    device_array = other.device_array;
    device_array_AD = other.device_array_AD;
    for (int device = 0; device < n_devices; ++device) {
      texture[device] = other.texture[device];
      texture_AD[device] = other.texture_AD[device];
    }
    check_cuda("TexLinearInterpolator1D::TexLinearInterpolator1D(const TexLinearInterpolator1D&)");
#endif
  }

  template <typename NT, typename Coordinates> TexLinearInterpolator1D<NT, Coordinates>::~TexLinearInterpolator1D()
  {
#ifdef __CUDACC__
    if (owner && m_data != nullptr) {
      for (int device = 0; device < n_devices; ++device) {
        cudaSetDevice(device);
        cudaDestroyTextureObject(texture[device]);
        cudaFreeArray(device_array[device]);
        check_cuda("TexLinearInterpolator1D::~TexLinearInterpolator1D");
      }
    }
    if (owner && m_data_AD != nullptr) {
      for (int device = 0; device < n_devices; ++device) {
        cudaSetDevice(device);
        cudaDestroyTextureObject(texture_AD[device]);
        cudaFreeArray(device_array_AD[device]);
        check_cuda("TexLinearInterpolator1D::~TexLinearInterpolator1D");
      }
    }
    cudaSetDevice(0);
#endif
  }

  template <typename NT, typename Coordinates> void TexLinearInterpolator1D<NT, Coordinates>::update()
  {
    if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

#ifdef __CUDACC__
    for (int device = 0; device < n_devices; ++device) {
      cudaSetDevice(device);

      cudaMemcpy2DToArray(device_array[device], 0, 0, &m_data[0], sizeof(m_data[0]) * size, sizeof(m_data[0]) * size, 1,
                          cudaMemcpyHostToDevice);
      if constexpr (std::is_same_v<ReturnType, autodiff::real>)
        cudaMemcpy2DToArray(device_array_AD[device], 0, 0, &m_data_AD[0], sizeof(m_data_AD[0]) * size,
                            sizeof(m_data_AD[0]) * size, 1, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      check_cuda("TexLinearInterpolator1D::update");
    }
    cudaSetDevice(0);
#endif
  }

  template <typename NT, typename Coordinates> float *TexLinearInterpolator1D<NT, Coordinates>::data() const
  {
    return m_data.get();
  }

  template <typename NT, typename Coordinates> float *TexLinearInterpolator1D<NT, Coordinates>::data_AD() const
  {
    return m_data_AD.get();
  }

  template <typename NT, typename Coordinates>
  typename TexLinearInterpolator1D<NT, Coordinates>::ReturnType &
  TexLinearInterpolator1D<NT, Coordinates>::operator[](const uint i)
  {
    if constexpr (std::is_same_v<ReturnType, autodiff::real>)
      throw std::runtime_error("Cannot access autodiff::real data directly");
    else if constexpr (std::is_same_v<ReturnType, float>)
      return m_data[i];
  }

  template <typename NT, typename Coordinates>
  const typename TexLinearInterpolator1D<NT, Coordinates>::ReturnType &
  TexLinearInterpolator1D<NT, Coordinates>::operator[](const uint i) const
  {
    if constexpr (std::is_same_v<ReturnType, autodiff::real>)
      throw std::runtime_error("Cannot access autodiff::real data directly");
    else if constexpr (std::is_same_v<ReturnType, float>)
      return m_data[i];
  }

  template class TexLinearInterpolator1D<double, LinearCoordinates1D<float>>;
  template class TexLinearInterpolator1D<float, LinearCoordinates1D<float>>;
  template class TexLinearInterpolator1D<autodiff::real, LinearCoordinates1D<float>>;
  template class TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>>;
  template class TexLinearInterpolator1D<float, LogarithmicCoordinates1D<float>>;
  template class TexLinearInterpolator1D<autodiff::real, LogarithmicCoordinates1D<float>>;

  template class TexLinearInterpolator1D<double, LinearCoordinates1D<double>>;
  template class TexLinearInterpolator1D<float, LinearCoordinates1D<double>>;
  template class TexLinearInterpolator1D<autodiff::real, LinearCoordinates1D<double>>;
  template class TexLinearInterpolator1D<double, LogarithmicCoordinates1D<double>>;
  template class TexLinearInterpolator1D<float, LogarithmicCoordinates1D<double>>;
  template class TexLinearInterpolator1D<autodiff::real, LogarithmicCoordinates1D<double>>;
} // namespace DiFfRG
