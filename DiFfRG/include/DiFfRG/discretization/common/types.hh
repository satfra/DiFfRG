#pragma once

// std
#include <type_traits>

namespace DiFfRG
{
  namespace internal
  {
    /// The mesh type declares `is_rectangular = true`.
    template <typename T>
    concept MeshIsRectangular = requires(T x) { T::is_rectangular; } && T::is_rectangular;

    /// T is a model (has `using Components = ...`) rather than a bare ComponentDescriptor.
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

    /// The Discretization was built from a model (not from a bare ComponentDescriptor).
    template <typename D>
    concept CarriesModel = requires { typename D::Model; } && !std::is_void_v<typename D::Model>;

    /// The model of a Discretization, used as the assembler's default Model; a clear error if it has none.
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
