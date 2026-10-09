#pragma once

#include <DiFfRG/discretization/FV/assembler/reconstruction_cache.hh>

#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      // The pre-assembly hook is called separately for residual and jacobian assembly.
      // A model must not assume one hook call per solver step or timestep.
      enum class AssemblyStage { residual, jacobian };

      template <int dim_, typename NumberType_, std::size_t n_components_> class FaceAssemblyView
      {
      public:
        static constexpr int dim = dim_;
        static constexpr std::size_t n_components = n_components_;
        using NumberType = NumberType_;
        using ReconstructionState = internal::FaceReconstructionState<dim, NumberType, n_components>;

        FaceAssemblyView(const dealii::Point<dim> &point, const dealii::Tensor<1, dim> &normal,
                         const double jxw, const bool boundary, const unsigned int cell_index,
                         const unsigned int face_index, const unsigned int neighbor_cell_index,
                         const ReconstructionState &reconstruction)
            : point_(point), normal_(normal), jxw_(jxw), boundary_(boundary), cell_index_(cell_index),
              face_index_(face_index), neighbor_cell_index_(neighbor_cell_index), reconstruction_(reconstruction)
        {
        }

        const dealii::Point<dim> &point() const { return point_; }
        const dealii::Tensor<1, dim> &normal() const { return normal_; }
        double jxw() const { return jxw_; }
        bool at_boundary() const { return boundary_; }
        unsigned int cell_index() const { return cell_index_; }
        unsigned int face_index() const { return face_index_; }
        unsigned int neighbor_cell_index() const { return neighbor_cell_index_; }
        const ReconstructionState &reconstruction() const { return reconstruction_; }

      private:
        dealii::Point<dim> point_;
        dealii::Tensor<1, dim> normal_;
        double jxw_ = 0.;
        bool boundary_ = false;
        unsigned int cell_index_ = 0;
        unsigned int face_index_ = 0;
        unsigned int neighbor_cell_index_ = std::numeric_limits<unsigned int>::max();
        const ReconstructionState &reconstruction_;
      };

      template <int dim_, typename NumberType_, std::size_t n_components_> class CellAssemblyView
      {
      public:
        static constexpr int dim = dim_;
        static constexpr std::size_t n_components = n_components_;
        using NumberType = NumberType_;
        using CellStencilData = internal::CellStencilData<dim, NumberType, n_components>;

        CellAssemblyView(const dealii::Point<dim> &point, const unsigned int cell_index,
                         const CellStencilData &stencil)
            : point_(point), cell_index_(cell_index), stencil_(stencil)
        {
        }

        const dealii::Point<dim> &point() const { return point_; }
        unsigned int cell_index() const { return cell_index_; }
        const CellStencilData &stencil() const { return stencil_; }

      private:
        dealii::Point<dim> point_;
        unsigned int cell_index_ = 0;
        const CellStencilData &stencil_;
      };

      template <typename Range>
      concept HasAssemblyViewRange =
          requires {
            typename Range::Context;
            typename Range::NumberType;
            typename Range::const_iterator;
            {
              Range::dim
            } -> std::convertible_to<int>;
            {
              Range::n_components
            } -> std::convertible_to<std::size_t>;
          } &&
          requires(const Range &range, const std::size_t i) {
            {
              range.begin()
            } -> std::same_as<typename Range::const_iterator>;
            {
              range.end()
            } -> std::same_as<typename Range::const_iterator>;
            {
              range.size()
            } -> std::convertible_to<std::size_t>;
            {
              range[i]
            } -> std::same_as<typename Range::Context>;
          };

      template <typename Range>
      concept HasFaceAssemblyViewRange =
          HasAssemblyViewRange<Range> && requires { typename Range::FaceAssemblyView; } &&
          std::same_as<typename Range::Context, typename Range::FaceAssemblyView>;

      template <typename Range>
      concept HasCellAssemblyViewRange =
          HasAssemblyViewRange<Range> && requires { typename Range::CellAssemblyView; } &&
          std::same_as<typename Range::Context, typename Range::CellAssemblyView>;

      template <int dim_, typename NumberType_, std::size_t n_components_, typename Iterator, typename GeometryProvider>
      class FaceAssemblyViewRange
      {
      public:
        static constexpr int dim = dim_;
        static constexpr std::size_t n_components = n_components_;
        using NumberType = NumberType_;
        using FaceAssemblyView = KurganovTadmor::FaceAssemblyView<dim, NumberType, n_components>;
        using Context = FaceAssemblyView;
        using SolutionReconstructionCache = internal::SolutionReconstructionCache<dim, NumberType, n_components>;

        class const_iterator
        {
        public:
          using iterator_category = std::random_access_iterator_tag;
          using difference_type = std::ptrdiff_t;
          using value_type = Context;
          using reference = value_type;
          using pointer = void;

          const_iterator() = default;
          const_iterator(const FaceAssemblyViewRange *range, const std::size_t index) : range_(range), index_(index) {}

          value_type operator*() const { return (*range_)[index_]; }
          value_type operator[](const difference_type offset) const
          {
            return (*range_)[static_cast<std::size_t>(static_cast<difference_type>(index_) + offset)];
          }

          const_iterator &operator++()
          {
            ++index_;
            return *this;
          }
          const_iterator operator++(int)
          {
            auto copy = *this;
            ++(*this);
            return copy;
          }
          const_iterator &operator--()
          {
            --index_;
            return *this;
          }
          const_iterator operator--(int)
          {
            auto copy = *this;
            --(*this);
            return copy;
          }
          const_iterator &operator+=(const difference_type offset)
          {
            index_ = static_cast<std::size_t>(static_cast<difference_type>(index_) + offset);
            return *this;
          }
          const_iterator &operator-=(const difference_type offset)
          {
            return *this += -offset;
          }

          friend const_iterator operator+(const const_iterator iterator, const difference_type offset)
          {
            auto copy = iterator;
            copy += offset;
            return copy;
          }
          friend const_iterator operator+(const difference_type offset, const const_iterator iterator)
          {
            return iterator + offset;
          }
          friend const_iterator operator-(const const_iterator iterator, const difference_type offset)
          {
            auto copy = iterator;
            copy -= offset;
            return copy;
          }
          friend difference_type operator-(const const_iterator &left, const const_iterator &right)
          {
            return static_cast<difference_type>(left.index_) - static_cast<difference_type>(right.index_);
          }

          friend bool operator==(const const_iterator &left, const const_iterator &right)
          {
            return left.range_ == right.range_ && left.index_ == right.index_;
          }
          friend bool operator!=(const const_iterator &left, const const_iterator &right) { return !(left == right); }
          friend bool operator<(const const_iterator &left, const const_iterator &right)
          {
            return left.index_ < right.index_;
          }
          friend bool operator>(const const_iterator &left, const const_iterator &right) { return right < left; }
          friend bool operator<=(const const_iterator &left, const const_iterator &right) { return !(right < left); }
          friend bool operator>=(const const_iterator &left, const const_iterator &right) { return !(left < right); }

        private:
          const FaceAssemblyViewRange *range_ = nullptr;
          std::size_t index_ = 0;
        };

        FaceAssemblyViewRange(const Iterator &begin, const Iterator &end, const SolutionReconstructionCache &cache,
                              GeometryProvider geometry_provider)
            : cache_(cache), geometry_provider_(std::move(geometry_provider))
        {
          for (auto cell = begin; cell != end; ++cell) {
            const auto cell_index = cell->active_cell_index();
            for (const auto face_index : cell->face_indices()) {
              const bool boundary = cell->at_boundary(face_index);
              unsigned int neighbor_index = std::numeric_limits<unsigned int>::max();
              if (!boundary) {
                neighbor_index = cell->neighbor(face_index)->active_cell_index();
                if (neighbor_index < cell_index) continue;
              }
              descriptors_.push_back({cell, face_index, boundary, neighbor_index});
            }
          }
        }

        const_iterator begin() const { return const_iterator(this, 0); }
        const_iterator end() const { return const_iterator(this, descriptors_.size()); }
        std::size_t size() const { return descriptors_.size(); }

        Context operator[](const std::size_t index) const
        {
          const auto &descriptor = descriptors_[index];
          const auto cell_index = descriptor.cell->active_cell_index();
          const auto &reconstruction = cache_.face_reconstructions[cell_index][descriptor.face_index];
          return Context(descriptor.cell->face(descriptor.face_index)->center(),
                         geometry_provider_.normal(descriptor.cell, descriptor.face_index),
                         geometry_provider_.jxw(descriptor.cell, descriptor.face_index), descriptor.boundary,
                         cell_index, descriptor.face_index, descriptor.neighbor_cell_index, reconstruction);
        }

      private:
        struct Descriptor {
          Iterator cell;
          unsigned int face_index = 0;
          bool boundary = false;
          unsigned int neighbor_cell_index = std::numeric_limits<unsigned int>::max();
        };

        std::vector<Descriptor> descriptors_;
        const SolutionReconstructionCache &cache_;
        GeometryProvider geometry_provider_;
      };

      template <int dim_, typename NumberType_, std::size_t n_components_, typename Iterator>
      class CellAssemblyViewRange
      {
      public:
        static constexpr int dim = dim_;
        static constexpr std::size_t n_components = n_components_;
        using NumberType = NumberType_;
        using CellAssemblyView = KurganovTadmor::CellAssemblyView<dim, NumberType, n_components>;
        using Context = CellAssemblyView;
        using SolutionReconstructionCache = internal::SolutionReconstructionCache<dim, NumberType, n_components>;

        class const_iterator
        {
        public:
          using iterator_category = std::random_access_iterator_tag;
          using difference_type = std::ptrdiff_t;
          using value_type = Context;
          using reference = value_type;
          using pointer = void;

          const_iterator() = default;
          const_iterator(const CellAssemblyViewRange *range, const std::size_t index) : range_(range), index_(index) {}

          value_type operator*() const { return (*range_)[index_]; }
          value_type operator[](const difference_type offset) const
          {
            return (*range_)[static_cast<std::size_t>(static_cast<difference_type>(index_) + offset)];
          }

          const_iterator &operator++()
          {
            ++index_;
            return *this;
          }
          const_iterator operator++(int)
          {
            auto copy = *this;
            ++(*this);
            return copy;
          }
          const_iterator &operator--()
          {
            --index_;
            return *this;
          }
          const_iterator operator--(int)
          {
            auto copy = *this;
            --(*this);
            return copy;
          }
          const_iterator &operator+=(const difference_type offset)
          {
            index_ = static_cast<std::size_t>(static_cast<difference_type>(index_) + offset);
            return *this;
          }
          const_iterator &operator-=(const difference_type offset)
          {
            return *this += -offset;
          }

          friend const_iterator operator+(const const_iterator iterator, const difference_type offset)
          {
            auto copy = iterator;
            copy += offset;
            return copy;
          }
          friend const_iterator operator+(const difference_type offset, const const_iterator iterator)
          {
            return iterator + offset;
          }
          friend const_iterator operator-(const const_iterator iterator, const difference_type offset)
          {
            auto copy = iterator;
            copy -= offset;
            return copy;
          }
          friend difference_type operator-(const const_iterator &left, const const_iterator &right)
          {
            return static_cast<difference_type>(left.index_) - static_cast<difference_type>(right.index_);
          }

          friend bool operator==(const const_iterator &left, const const_iterator &right)
          {
            return left.range_ == right.range_ && left.index_ == right.index_;
          }
          friend bool operator!=(const const_iterator &left, const const_iterator &right) { return !(left == right); }
          friend bool operator<(const const_iterator &left, const const_iterator &right)
          {
            return left.index_ < right.index_;
          }
          friend bool operator>(const const_iterator &left, const const_iterator &right) { return right < left; }
          friend bool operator<=(const const_iterator &left, const const_iterator &right) { return !(right < left); }
          friend bool operator>=(const const_iterator &left, const const_iterator &right) { return !(left < right); }

        private:
          const CellAssemblyViewRange *range_ = nullptr;
          std::size_t index_ = 0;
        };

        CellAssemblyViewRange(const Iterator &begin, const Iterator &end, const SolutionReconstructionCache &cache)
            : cache_(cache)
        {
          for (auto cell = begin; cell != end; ++cell)
            descriptors_.push_back({cell, cell->active_cell_index()});
        }

        const_iterator begin() const { return const_iterator(this, 0); }
        const_iterator end() const { return const_iterator(this, descriptors_.size()); }
        std::size_t size() const { return descriptors_.size(); }

        Context operator[](const std::size_t index) const
        {
          const auto &descriptor = descriptors_[index];
          return Context(descriptor.cell->center(), descriptor.cell_index, cache_.cell_stencils[descriptor.cell_index]);
        }

      private:
        struct Descriptor {
          Iterator cell;
          unsigned int cell_index = 0;
        };

        std::vector<Descriptor> descriptors_;
        const SolutionReconstructionCache &cache_;
      };

      template <typename FaceRange, typename CellRange> class AssemblyContextView
      {
      public:
        static_assert(FaceRange::dim == CellRange::dim);
        static_assert(FaceRange::n_components == CellRange::n_components);

        static constexpr int dim = FaceRange::dim;
        static constexpr std::size_t n_components = FaceRange::n_components;
        using NumberType = typename FaceRange::NumberType;
        using FaceAssemblyViewRange = FaceRange;
        using CellAssemblyViewRange = CellRange;

        AssemblyContextView(FaceRange faces, CellRange cells) : faces_(std::move(faces)), cells_(std::move(cells)) {}

        const FaceRange &faces() const { return faces_; }
        const CellRange &cells() const { return cells_; }

      private:
        FaceRange faces_;
        CellRange cells_;
      };

      template <typename Context>
      concept HasAssemblyContextView =
          requires(const Context &context) {
            typename Context::FaceAssemblyViewRange;
            typename Context::CellAssemblyViewRange;
            typename Context::NumberType;
            {
              Context::dim
            } -> std::convertible_to<int>;
            {
              Context::n_components
            } -> std::convertible_to<std::size_t>;
            {
              context.faces()
            } -> std::same_as<const typename Context::FaceAssemblyViewRange &>;
            {
              context.cells()
            } -> std::same_as<const typename Context::CellAssemblyViewRange &>;
          } && HasFaceAssemblyViewRange<typename Context::FaceAssemblyViewRange> &&
          HasCellAssemblyViewRange<typename Context::CellAssemblyViewRange>;

      // Assembly context views are non-owning views over the per-solution reconstruction cache and
      // active-cell iterators. They are valid only for the synchronous fv_kt_pre_assembly call.
      template <typename Model, typename Context>
      concept HasFVKTAssemblyHook =
          HasAssemblyContextView<Context> && requires(Model &model, const AssemblyStage stage, const Context &context) {
            {
              model.fv_kt_pre_assembly(stage, context)
            } -> std::same_as<void>;
          };

      template <typename Model, HasAssemblyContextView Context>
      void dispatch_fv_kt_pre_assembly(Model &model, const AssemblyStage stage, const Context &context)
      {
        if constexpr (HasFVKTAssemblyHook<Model, Context>) model.fv_kt_pre_assembly(stage, context);
      }
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
