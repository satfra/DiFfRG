#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>

// external libraries
#include <autodiff/forward/real.hpp>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_direct.h>

namespace DiFfRG
{
  template <typename T>
  concept IsContainer = requires(T x) { x[0]; };

  namespace get_type
  {
    namespace internal
    {
      //--------------------------------------------------
      // Hidden unspecified type helpers
      //--------------------------------------------------

      template <typename VectorType> struct _NumberType;

      template <typename SparseMatrixType> struct _SparsityPattern;

      template <typename SparseMatrixType> struct _InverseSparseMatrixType;

      //--------------------------------------------------
      // Specified type helpers for standard vectors
      //--------------------------------------------------

      template <typename NT> struct _NumberType<dealii::Vector<NT>> {
        using value = NT;
      };
      template <typename NT> struct _SparsityPattern<dealii::SparseMatrix<NT>> {
        using value = dealii::SparsityPattern;
      };
      template <typename NT> struct _InverseSparseMatrixType<dealii::SparseMatrix<NT>> {
        using value = dealii::SparseDirectUMFPACK;
      };

      //--------------------------------------------------
      // Specified type helpers for block vectors
      //--------------------------------------------------

      template <typename NT> struct _NumberType<dealii::BlockVector<NT>> {
        using value = NT;
      };
      template <typename NT> struct _SparsityPattern<dealii::BlockSparseMatrix<NT>> {
        using value = dealii::BlockSparsityPattern;
      };
      template <typename NT> struct _InverseSparseMatrixType<dealii::BlockSparseMatrix<NT>> {
        using value = dealii::SparseDirectUMFPACK;
      };
    } // namespace internal

    template <typename VectorType> using NumberType = typename internal::_NumberType<VectorType>::value;

    template <typename SparseMatrixType>
    using SparsityPattern = typename internal::_SparsityPattern<SparseMatrixType>::value;

    template <typename SparseMatrixType>
    using InverseSparseMatrixType = typename internal::_InverseSparseMatrixType<SparseMatrixType>::value;

    namespace internal
    {
      template <typename CT> struct _ctype;

      template <> struct _ctype<float> {
        using value = float;
      };

      template <> struct _ctype<double> {
        using value = double;
      };

      template <> struct _ctype<complex<float>> {
        using value = float;
      };

      template <> struct _ctype<complex<double>> {
        using value = double;
      };

      template <> struct _ctype<autodiff::Real<1, float>> {
        using value = float;
      };

      template <> struct _ctype<autodiff::Real<1, double>> {
        using value = double;
      };

      template <> struct _ctype<autodiff::Real<1, complex<float>>> {
        using value = float;
      };

      template <> struct _ctype<autodiff::Real<1, complex<double>>> {
        using value = double;
      };
    } // namespace internal

    template <typename CT> using ctype = typename internal::_ctype<CT>::value;
  } // namespace get_type
} // namespace DiFfRG