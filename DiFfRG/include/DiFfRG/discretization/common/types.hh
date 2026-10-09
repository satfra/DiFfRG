#pragma once

// std
#include <type_traits>

namespace DiFfRG
{
  template <typename T>
  concept MeshIsRectangular = requires(T x) { T::is_rectangular; } && T::is_rectangular;

  namespace internal
  {
    /**
     * @brief Whether a type is a model, i.e. carries the component descriptor rather than being one.
     *
     * Models expose `using Components = ...`; a bare ComponentDescriptor does not (it exposes
     * FEFunction_Descriptor / Variable_Descriptor / Extractor_Descriptor instead). That asymmetry is
     * what lets a Discretization take either one as its first argument.
     */
    template <typename T>
    concept IsModel = requires { typename T::Components; };

    /// The component descriptor of a model, or the descriptor itself if one was passed directly.
    template <typename T> struct components_of {
      using type = T;
    };
    template <IsModel T> struct components_of<T> {
      using type = typename T::Components;
    };

    /// The model a Discretization was built from, or void if it was built from a bare descriptor.
    template <typename T> struct model_of_descriptor {
      using type = void;
    };
    template <IsModel T> struct model_of_descriptor<T> {
      using type = T;
    };

    /**
     * @brief Whether a Discretization knows which model it belongs to.
     *
     * False exactly for the discretizations built from a bare ComponentDescriptor, which exist in
     * tests that exercise constraints and vector layout without ever building an assembler.
     */
    template <typename D>
    concept CarriesModel = requires { typename D::Model; } && !std::is_void_v<typename D::Model>;

    /**
     * @brief The model an assembler should use when the application did not name one.
     *
     * The static_assert is what turns "Discretization::Model is void" into one readable line. It
     * fires only when the default argument is actually used, so naming the model explicitly --
     * `CG::Assembler<Discretization, MyModel>` -- bypasses it entirely.
     */
    template <typename D> struct assembler_model_of {
      static_assert(CarriesModel<D>,
                    "This Discretization was built from a bare ComponentDescriptor and so does not know "
                    "its Model. Either name the model explicitly, e.g. CG::Assembler<Discretization, MyModel>, "
                    "or build the Discretization from the model instead of its components, e.g. "
                    "CG::Discretization<MyModel, RectangularMesh<dim>>.");
      // Never void: keeps the assembler base from forming a `void &` member and burying the
      // message above under a wall of follow-on errors.
      using type = std::conditional_t<CarriesModel<D>, typename D::Model, D>;
    };
  } // namespace internal
} // namespace DiFfRG
