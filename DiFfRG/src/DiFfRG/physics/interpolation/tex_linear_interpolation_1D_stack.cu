// DiFfRG
#include <DiFfRG/discretization/grid/combined_coordinates.hh>
#include <DiFfRG/physics/interpolation/tex_linear_interpolation_1D_stack.hh>

namespace DiFfRG
{
  template <typename NT, typename Coordinates, uint max_stack_size>
  TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::TexLinearInterpolator1DStack(
      const std::vector<NT> &data, const Coordinates &coordinates)
      : TexLinearInterpolator1DStack(data.data(), coordinates)
  {
    if (data.size() != coordinates.size()) throw std::runtime_error("Data and coordinates must have the same size");
  }

  template <typename NT, typename Coordinates, uint max_stack_size>
  TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::TexLinearInterpolator1DStack(
      const Coordinates &coordinates)
      : TexLinearInterpolator1DStack(std::vector<NT>(coordinates.size(), NT(0)), coordinates)
  {
  }

  template <typename NT, typename Coordinates, uint max_stack_size>
  TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::TexLinearInterpolator1DStack(
      const NT *in_data, const Coordinates &coordinates)
      : stack_size(coordinates.sizes()[0]), p_size(coordinates.sizes()[1]), coordinates(coordinates), owner(true)
  {
    if (stack_size > max_stack_size)
      throw std::runtime_error("TexLinearInterpolator1DStack: Stack size (" + std::to_string(stack_size) +
                               ") exceeds maximum stack size (" + std::to_string(max_stack_size) + ")");

    device_array.resize(stack_size, nullptr);
    device_array_AD.resize(stack_size, nullptr);

    for (uint s = 0; s < stack_size; ++s) {
      if constexpr (std::is_same_v<float, ReturnType>) {
        // Copy input data and possibly cast to float
        m_data.emplace_back(std::shared_ptr<float[]>(new float[p_size]));
        for (uint i = 0; i < p_size; ++i)
          m_data[s][i] = static_cast<float>(in_data[s * p_size + i]);
#ifdef __CUDACC__
        // Allocate device array and copy data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&device_array[s], &channelDesc, p_size);
        cudaMemcpy2DToArray(device_array[s], 0, 0, &m_data[s][0], sizeof(m_data[s][0]) * p_size,
                            sizeof(m_data[s][0]) * p_size, 1, cudaMemcpyHostToDevice);

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = device_array[s];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaCreateTextureObject(&texture[s], &resDesc, &texDesc, NULL);
        cudaDeviceSynchronize();
        check_cuda("TexLinearInterpolator1DStack::TexLinearInterpolator1DStack");
#endif
      } else if constexpr (std::is_same_v<autodiff::real, ReturnType>) {
        // Copy input data and possibly cast to float
        m_data.emplace_back(std::shared_ptr<float[]>(new float[p_size]));
        m_data_AD.emplace_back(std::shared_ptr<float[]>(new float[p_size]));
        for (uint i = 0; i < p_size; ++i) {
          m_data[s][i] = static_cast<float>(val(in_data[s * p_size + i]));
          m_data_AD[s][i] = static_cast<float>(derivative(in_data[s * p_size + i]));
        }

#ifdef __CUDACC__
        // Allocate device array and copy data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&device_array[s], &channelDesc, p_size);
        cudaMallocArray(&device_array_AD[s], &channelDesc, p_size);
        cudaMemcpy2DToArray(device_array[s], 0, 0, &m_data[s][0], sizeof(m_data[s][0]) * p_size,
                            sizeof(m_data[s][0]) * p_size, 1, cudaMemcpyHostToDevice);
        cudaMemcpy2DToArray(device_array_AD[s], 0, 0, &m_data_AD[s][0], sizeof(m_data_AD[s][0]) * p_size,
                            sizeof(m_data_AD[s][0]) * p_size, 1, cudaMemcpyHostToDevice);

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = device_array[s];

        struct cudaResourceDesc resDesc_AD;
        memset(&resDesc_AD, 0, sizeof(resDesc_AD));
        resDesc_AD.resType = cudaResourceTypeArray;
        resDesc_AD.res.array.array = device_array_AD[s];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaCreateTextureObject(&texture[s], &resDesc, &texDesc, NULL);
        cudaCreateTextureObject(&texture_AD[s], &resDesc_AD, &texDesc, NULL);
        cudaDeviceSynchronize();
        check_cuda("TexLinearInterpolator1DStack::TexLinearInterpolator1DStack");
#endif
      }
    }
  }

  template <typename NT, typename Coordinates, uint max_stack_size>
  TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::TexLinearInterpolator1DStack(
      const TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size> &other)
      : stack_size(other.stack_size), p_size(other.p_size), coordinates(other.coordinates), owner(false)
  {
    m_data = other.m_data;
    m_data_AD = other.m_data_AD;
#ifdef __CUDACC__
    device_array = other.device_array;
    for (uint s = 0; s < stack_size; ++s)
      texture[s] = other.texture[s];
    device_array_AD = other.device_array_AD;
    for (uint s = 0; s < stack_size; ++s)
      texture_AD[s] = other.texture_AD[s];
    check_cuda("TexLinearInterpolator1DStack::TexLinearInterpolator1DStack(const TexLinearInterpolator1DStack&)");
#endif
  }

  template <typename NT, typename Coordinates, uint max_stack_size>
  TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::~TexLinearInterpolator1DStack()
  {
#ifdef __CUDACC__
    for (uint s = 0; s < stack_size; ++s) {
      if (owner && m_data.size() > s) {
        cudaDestroyTextureObject(texture[s]);
        cudaFreeArray(device_array[s]);
        check_cuda("TexLinearInterpolator1DStack::~TexLinearInterpolator1DStack");
      }
      if (owner && m_data_AD.size() > s) {
        cudaDestroyTextureObject(texture_AD[s]);
        cudaFreeArray(device_array_AD[s]);
        check_cuda("TexLinearInterpolator1DStack::~TexLinearInterpolator1DStack");
      }
    }
#endif
  }

  template <typename NT, typename Coordinates, uint max_stack_size>
  void TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::update()
  {
    if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

#ifdef __CUDACC__
    for (uint s = 0; s < stack_size; ++s) {
      cudaMemcpy2DToArray(device_array[s], 0, 0, &m_data[s][0], sizeof(m_data[s][0]) * p_size,
                          sizeof(m_data[s][0]) * p_size, 1, cudaMemcpyHostToDevice);
      if constexpr (std::is_same_v<ReturnType, autodiff::real>)
        cudaMemcpy2DToArray(device_array_AD[s], 0, 0, &m_data_AD[s][0], sizeof(m_data_AD[s][0]) * p_size,
                            sizeof(m_data_AD[s][0]) * p_size, 1, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      check_cuda("TexLinearInterpolator1DStack::update");
    }
#endif
  }

  template <typename NT, typename Coordinates, uint max_stack_size>
  float *TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::data(const uint s) const
  {
    return m_data[s].get();
  }

  template <typename NT, typename Coordinates, uint max_stack_size>
  float *TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::data_AD(const uint s) const
  {
    return m_data_AD[s].get();
  }

  template <typename NT, typename Coordinates, uint max_stack_size>
  typename TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::ReturnType &
  TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::operator[](const uint i)
  {
    if constexpr (std::is_same_v<ReturnType, autodiff::real>)
      throw std::runtime_error("Cannot access autodiff::real data directly");
    else if constexpr (std::is_same_v<ReturnType, float>)
      return m_data[i / stack_size][i % stack_size];
  }

  template <typename NT, typename Coordinates, uint max_stack_size>
  const typename TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::ReturnType &
  TexLinearInterpolator1DStack<NT, Coordinates, max_stack_size>::operator[](const uint i) const
  {
    if constexpr (std::is_same_v<ReturnType, autodiff::real>)
      throw std::runtime_error("Cannot access autodiff::real data directly");
    else if constexpr (std::is_same_v<ReturnType, float>)
      return m_data[i / stack_size][i % stack_size];
  }

  template class TexLinearInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, float>>;
  template class TexLinearInterpolator1DStack<float, BosonicCoordinates1DFiniteT<int, float>>;
  template class TexLinearInterpolator1DStack<autodiff::real, BosonicCoordinates1DFiniteT<int, float>>;

  template class TexLinearInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double>>;
  template class TexLinearInterpolator1DStack<float, BosonicCoordinates1DFiniteT<int, double>>;
  template class TexLinearInterpolator1DStack<autodiff::real, BosonicCoordinates1DFiniteT<int, double>>;

  template class TexLinearInterpolator1DStack<double, FermionicCoordinates1DFiniteT<int, float>>;
  template class TexLinearInterpolator1DStack<float, FermionicCoordinates1DFiniteT<int, float>>;
  template class TexLinearInterpolator1DStack<autodiff::real, FermionicCoordinates1DFiniteT<int, float>>;

  template class TexLinearInterpolator1DStack<double, FermionicCoordinates1DFiniteT<int, double>>;
  template class TexLinearInterpolator1DStack<float, FermionicCoordinates1DFiniteT<int, double>>;
  template class TexLinearInterpolator1DStack<autodiff::real, FermionicCoordinates1DFiniteT<int, double>>;
} // namespace DiFfRG
