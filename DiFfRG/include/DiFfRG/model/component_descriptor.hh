#pragma once

// standard library
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// external libraries
#include <deal.II/dofs/dof_tools.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  /**
   * @brief A class to describe a function with a compile-time name and a fixed number of dimensions.
   *
   * @tparam _str The name of the function.
   * @tparam _val The dimensions of the function.
   */
  template <FixedString _str, size_t... _val> struct FunctionND {
    static constexpr FixedString name = _str;
    static constexpr size_t size = (_val * ...);
    static constexpr size_t dim = sizeof...(_val);
    static constexpr std::array<size_t, dim> nd_sizes{{_val...}};
  };

  /*
   * @brief A class to describe a scalar variable.
   *
   * @tparam _str The name of the variable.
   */
  template <FixedString _str> struct Scalar {
    /*
    You're probably wondering why this is not a specialization of FunctionND.
    Well, depending on your compiler, this may end up in a wierd bug, where it messes up the internals of the array in
    FixedString. I was not able to fully debug this, and therefore we circumvent the problem by very explicitly defining
    the Scalar class.
    */
    static constexpr FixedString name = _str;
    static constexpr size_t size = 1;
    static constexpr size_t dim = 1;
    static constexpr std::array<size_t, dim> nd_sizes{{1}};
  };

  template <typename... _descriptors> struct SubDescriptor {
    // tuple of all FixedString names
    static constexpr auto names_tuple = std::tie(_descriptors::name...);

    static constexpr std::array<const char *, sizeof...(_descriptors)> names{{_descriptors::name...}};
    static constexpr std::array<size_t, sizeof...(_descriptors)> sizes{{_descriptors::size...}};
    using descriptor_tuple = std::tuple<_descriptors...>;

    static constexpr size_t total_size = (0 + ... + _descriptors::size);

    // If two names are the same, the program should not compile
    static_assert(
        []<size_t... I>(std::index_sequence<I...>) {
          if constexpr (sizeof...(_descriptors) > 1)
            for (size_t i : {I...})
              for (size_t j : {I...})
                if (i != j && internal::strings_equal(names[i], names[j])) return false;
          return true;
        }(std::make_index_sequence<sizeof...(_descriptors)>{}),
        "Names of a SubDescriptor must be unique!");

    template <unsigned N> consteval size_t get(FixedString<N> name) const
    {
      size_t running_sum = 0;
      for (size_t i = 0; i < names.size(); ++i) {
        if (internal::strings_equal(names[i], name)) return running_sum;
        running_sum += sizes[i];
      }
      // produce a compile-time error if the name is not found in the list
      throw std::invalid_argument(
          "SubDescriptor::get: Name \"" + std::string(name) +
          "\" not found. Available names are: " + ((std::string(_descriptors::name) + "; ") + ...));
    }

    template <unsigned N> consteval size_t operator[](FixedString<N> name) const { return get(name); }
    template <unsigned N> consteval size_t operator()(FixedString<N> name) const { return get(name); }

    template <unsigned N> consteval size_t operator[](char const (&arr)[N]) const
    {
      return get(FixedString<N - 1>(arr));
    }
    template <unsigned N> consteval size_t operator()(char const (&arr)[N]) const
    {
      return get(FixedString<N - 1>(arr));
    }

    consteval char const *name(size_t index) const { return names[index]; }

    consteval std::array<const char *, sizeof...(_descriptors)> get_names() const { return names; }
    /**
     * @brief One name per component: a descriptor of size n > 1 contributes name_0, ..., name_{n-1}.
     */
    static std::vector<std::string> get_names_vector()
    {
      std::vector<std::string> result;
      for (size_t i = 0; i < names.size(); ++i) {
        if (sizes[i] == 1)
          result.emplace_back(names[i]);
        else
          for (size_t j = 0; j < sizes[i]; ++j)
            result.push_back(std::string(names[i]) + "_" + std::to_string(j));
      }
      return result;
    }

    template <unsigned N> consteval size_t size(char const (&arr)[N]) const
    {
      for (size_t i = 0; i < names.size(); ++i)
        if (internal::strings_equal(names[i], arr)) return sizes[i];
      throw std::invalid_argument(
          "SubDescriptor::size: Name \"" + std::string(arr) +
          "\" not found. Available names are: " + ((std::string(_descriptors::name) + "; ") + ...));
    }
  };

  template <typename... descriptors> using FEFunctionDescriptor = SubDescriptor<descriptors...>;
  template <typename... descriptors> using VariableDescriptor = SubDescriptor<descriptors...>;
  template <typename... descriptors> using ExtractorDescriptor = SubDescriptor<descriptors...>;

  /**
   * @brief A class to describe how many FE functions, additional variables and extractors are used in a model.
   *
   * @tparam components_ A DiFfRG::NumberFEFunctions object, which describes how many FE functions are used.
   * @tparam variables_ A DiFfRG::NumberVariables object, which describes how many additional variables are used.
   * @tparam extractors_ A DiFfRG::NumberExtractors object, which describes how many extractors are used.
   */
  template <typename _FEFunctionDescriptor, typename _VariableDescriptor = VariableDescriptor<>,
            typename _ExtractorDescriptor = ExtractorDescriptor<>, typename... LDGDescriptors>
  class ComponentDescriptor
  {
    using CouplingList = std::vector<std::array<uint, 2>>;

  public:
    using FEFunction_Descriptor = _FEFunctionDescriptor;
    using Variable_Descriptor = _VariableDescriptor;
    using Extractor_Descriptor = _ExtractorDescriptor;

  private:
    static constexpr uint n_fe_subsystems = sizeof...(LDGDescriptors) + 1;
    static constexpr std::array<uint, n_fe_subsystems> n_components{
        {FEFunction_Descriptor::total_size, LDGDescriptors::total_size...}};
    static constexpr uint n_variables = Variable_Descriptor::total_size;
    static constexpr uint n_extractors = Extractor_Descriptor::total_size;

  public:
    ComponentDescriptor()
    {
      for (auto &dep : j_const)
        for (auto &indep : dep)
          indep = false;
    }

    /**
     * @brief LDG: declare that component @p dependent of level @p dependent_subsystem is built from component
     * @p independent of level @p independent_subsystem = dependent_subsystem - 1. Level 0 holds the FE functions.
     *
     * The couplings between non-adjacent levels follow by composition, whatever the order of the calls.
     */
    void add_dependency(uint dependent_subsystem, uint dependent, uint independent_subsystem, uint independent)
    {
      check_adjacent_levels(dependent_subsystem, independent_subsystem, "add_dependency");
      auto &adjacent = couplings[dependent_subsystem][independent_subsystem];
      const std::array<uint, 2> coupling{{dependent, independent}};
      if (std::find(adjacent.begin(), adjacent.end(), coupling) == adjacent.end()) adjacent.push_back(coupling);

      // Level dep couples to level dep - 2, dep - 3, ... through the adjacent couplings in between.
      for (uint dep = 2; dep < n_fe_subsystems; ++dep)
        for (int indep = int(dep) - 2; indep >= 0; --indep) {
          auto &composed = couplings[dep][indep];
          composed.clear();
          for (const auto &outer : couplings[dep][dep - 1])
            for (const auto &inner : couplings[dep - 1][indep])
              if (outer[1] == inner[0]) {
                const std::array<uint, 2> c{{outer[0], inner[1]}};
                if (std::find(composed.begin(), composed.end(), c) == composed.end()) composed.push_back(c);
              }
        }
    }

    /**
     * @brief LDG: declare that level @p dependent_subsystem is a linear, homogeneous function of level
     * @p independent_subsystem = dependent_subsystem - 1, i.e. \f$L_k = A\, L_{k-1}\f$ with a fixed matrix
     * \f$A\f$: ldg_flux and ldg_source linear in their argument, no constant terms, also not on the boundary. The
     * assembler then builds the matrix once. Wrong for anything else, see jacobians_constant.
     */
    void set_jacobian_constant(uint dependent_subsystem, uint independent_subsystem)
    {
      check_adjacent_levels(dependent_subsystem, independent_subsystem, "set_jacobian_constant");
      j_const[dependent_subsystem][independent_subsystem] = true;
    }

    /// The (dependent, independent) component pairs coupling the two levels, see add_dependency.
    const CouplingList &ldg_couplings(uint dependent_subsystem, uint independent_subsystem) const
    {
      return couplings[dependent_subsystem][independent_subsystem];
    }

    /**
     * @brief Whether level @p dependent_subsystem is a fixed linear map of level @p independent_subsystem <
     * dependent_subsystem, i.e. whether every level in between was declared constant with set_jacobian_constant.
     */
    bool jacobians_constant(uint dependent_subsystem, uint independent_subsystem) const
    {
      if (independent_subsystem >= dependent_subsystem || dependent_subsystem >= n_fe_subsystems) return false;
      for (uint level = independent_subsystem + 1; level <= dependent_subsystem; ++level)
        if (!j_const[level][level - 1]) return false;
      return true;
    }

    static constexpr uint count_fe_functions(uint subsystem = 0) { return n_components[subsystem]; }
    static constexpr uint count_variables() { return n_variables; }
    static constexpr uint count_extractors() { return n_extractors; }
    static constexpr uint count_fe_subsystems() { return n_fe_subsystems; }

    template <typename DoFH> static std::vector<uint> get_component_block_structure(const DoFH &dofh)
    {
      std::vector<uint> dofs_per_component = dealii::DoFTools::count_dofs_per_fe_component(dofh);
      return dofs_per_component;
    }

  private:
    static void check_adjacent_levels(uint dependent_subsystem, uint independent_subsystem, const char *caller)
    {
      if (dependent_subsystem >= n_fe_subsystems || independent_subsystem >= n_fe_subsystems)
        throw std::runtime_error(std::string(caller) + ": the subsystems do not exist.");
      if (dependent_subsystem != independent_subsystem + 1)
        throw std::runtime_error(std::string(caller) + ": only neighbouring levels can be coupled directly.");
    }

    template <typename T> using SubsystemMatrix = std::array<std::array<T, n_fe_subsystems>, n_fe_subsystems>;

    SubsystemMatrix<CouplingList> couplings;
    SubsystemMatrix<bool> j_const;
  };
} // namespace DiFfRG
