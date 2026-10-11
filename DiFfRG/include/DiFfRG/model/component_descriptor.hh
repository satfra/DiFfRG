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

/**
 * @file component_descriptor.hh
 * @brief Declaring the unknowns of a model: FE functions, variables, extractors and LDG levels, with named access.
 *
 * Each entry has a compile-time name and is a single number (Scalar) or an array (FunctionND). Entries are grouped
 * into an FEFunctionDescriptor, VariableDescriptor or ExtractorDescriptor, and these into a ComponentDescriptor,
 * which is the `Components` argument of def::AbstractModel:
 * @code
 * using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, FunctionND<"A", 3>>;  // u, A_0, A_1, A_2
 * using VariableDesc = VariableDescriptor<Scalar<"Z">>;
 * using ExtractorDesc = ExtractorDescriptor<Scalar<"eta">, Scalar<"m2">>;
 * using LDG1Desc = FEFunctionDescriptor<Scalar<"du">>;                              // optional LDG level 1
 * using Components = ComponentDescriptor<FEFunctionDesc, VariableDesc, ExtractorDesc, LDG1Desc>;
 *
 * constexpr auto idxf = FEFunctionDesc{};
 * constexpr auto idxv = VariableDesc{};
 * // in the model: u_i[idxf("u")], u_i[idxf("A") + 2], v_a[idxv["Z"]]
 * @endcode
 * The index lookup runs at compile time; an unknown name is a compile error, and names within one descriptor must be
 * unique. An array entry occupies consecutive indices starting at `idx("name")`.
 */

namespace DiFfRG
{
  /**
   * @brief An array-valued entry named @p _str with \f$n_1 \times n_2 \times \dots\f$ components, stored flat.
   *
   * @tparam _str the name.
   * @tparam _val the extents \f$n_1, n_2, \dots\f$.
   */
  template <FixedString _str, size_t... _val> struct FunctionND {
    static constexpr FixedString name = _str;
    static constexpr size_t size = (_val * ...);
    static constexpr size_t dim = sizeof...(_val);
    static constexpr std::array<size_t, dim> nd_sizes{{_val...}};
  };

  /**
   * @brief A single-number entry named @p _str.
   *
   * @tparam _str the name.
   */
  template <FixedString _str> struct Scalar {
    // Not an alias of FunctionND<_str, 1>: with some compilers that corrupts the FixedString name.
    static constexpr FixedString name = _str;
    static constexpr size_t size = 1;
    static constexpr size_t dim = 1;
    static constexpr std::array<size_t, dim> nd_sizes{{1}};
  };

  /**
   * @brief A list of named entries (Scalar, FunctionND) with compile-time index lookup. Use it through the aliases
   * FEFunctionDescriptor, VariableDescriptor, ExtractorDescriptor.
   */
  template <typename... _descriptors> struct SubDescriptor {
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

    /// Index of the first component of entry @p name; unknown names do not compile.
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

    /// Same as get(): `idx["u"]` and `idx("u")` are the index of entry "u".
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

    /// Name of the entry at position @p index in the list (not a component index).
    consteval char const *name(size_t index) const { return names[index]; }

    /// Names of the entries, one per entry.
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

    /// Number of components of entry @p arr.
    template <unsigned N> consteval size_t size(char const (&arr)[N]) const
    {
      for (size_t i = 0; i < names.size(); ++i)
        if (internal::strings_equal(names[i], arr)) return sizes[i];
      throw std::invalid_argument(
          "SubDescriptor::size: Name \"" + std::string(arr) +
          "\" not found. Available names are: " + ((std::string(_descriptors::name) + "; ") + ...));
    }
  };

  /// The FE functions (also the components of an LDG level), see component_descriptor.hh.
  template <typename... descriptors> using FEFunctionDescriptor = SubDescriptor<descriptors...>;
  /// The variables, see component_descriptor.hh.
  template <typename... descriptors> using VariableDescriptor = SubDescriptor<descriptors...>;
  /// The extractors, see component_descriptor.hh.
  template <typename... descriptors> using ExtractorDescriptor = SubDescriptor<descriptors...>;

  /**
   * @brief All unknowns of a model: FE functions, variables, extractors and the LDG levels; the `Components`
   * argument of def::AbstractModel. See component_descriptor.hh for an example.
   *
   * An object of it also stores the LDG coupling declarations (add_dependency, set_jacobian_constant), made in the
   * model's constructor through def::AbstractModel::components().
   *
   * @tparam _FEFunctionDescriptor an FEFunctionDescriptor: the FE functions (level 0).
   * @tparam _VariableDescriptor a VariableDescriptor; default none.
   * @tparam _ExtractorDescriptor an ExtractorDescriptor; default none.
   * @tparam LDGDescriptors one FEFunctionDescriptor per LDG level 1, 2, 3 (at most three); default none.
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
    /// No LDG couplings, nothing declared constant.
    ComponentDescriptor()
    {
      for (auto &dep : j_const)
        for (auto &indep : dep)
          indep = false;
    }

    /**
     * @brief LDG: component @p dependent of level @p dependent_subsystem depends on component @p independent of level
     * @p independent_subsystem. Level 0 holds the FE functions.
     *
     * Only adjacent levels (independent_subsystem = dependent_subsystem - 1) may be declared, otherwise this throws.
     * Couplings across several levels are composed automatically, whatever the order of the calls.
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
     * @brief LDG: level @p dependent_subsystem is a fixed linear, homogeneous map of level @p independent_subsystem =
     * dependent_subsystem - 1, i.e. \f$L_k = A\, L_{k-1}\f$ with a constant matrix \f$A\f$.
     *
     * ldg_flux and ldg_source must be linear in their argument, without constant terms, also in the boundary
     * fluxes. The assembler then builds \f$A\f$ once instead of reassembling, provided all levels below are also
     * constant (see jacobians_constant). The claim is checked on first use and throws if violated.
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

    /// Number of components of level @p subsystem (0: the FE functions).
    static constexpr uint count_fe_functions(uint subsystem = 0) { return n_components[subsystem]; }
    /// Number of variables.
    static constexpr uint count_variables() { return n_variables; }
    /// Number of extractors.
    static constexpr uint count_extractors() { return n_extractors; }
    /// Number of levels, 1 + the number of LDG levels.
    static constexpr uint count_fe_subsystems() { return n_fe_subsystems; }

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
