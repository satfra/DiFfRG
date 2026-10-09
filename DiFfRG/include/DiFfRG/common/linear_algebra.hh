#pragma once

// external libraries
#include <deal.II/base/config.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// The distributed linear algebra is PETSc-backed. deal.II ships these headers
// unconditionally but leaves them empty without PETSc, so guard on the feature
// macro (defined by deal.II/base/config.h above) rather than on their presence.
#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#endif

#ifdef DEAL_II_WITH_MPI
#include <deal.II/distributed/shared_tria.h>
#endif
#include <deal.II/grid/tria.h>

// std
#include <type_traits>

/**
 * @file linear_algebra.hh
 *
 * @brief All compile-time knowledge about which linear algebra types DiFfRG uses.
 *
 * This is the single place that knows the *types*: which vector belongs to which
 * matrix, which of them are distributed, and what an unqualified build gets by
 * default. It deliberately holds no runtime code -- the MPI-collective helpers that
 * branch on this knowledge live in discretization/common/la_policy.hh, and the
 * linear-solver selection lives in timestepping/default_linear_solver.hh (solvers
 * sit above this layer, so naming them here would invert the include order).
 *
 * Split out of common/types.hh, which now keeps only the generic C++ concepts. That
 * matters beyond tidiness: everything under physics/ needs `get_type::ctype` and
 * nothing else, yet used to pull sparse_direct.h, block_sparse_matrix.h and four
 * PETSc headers into every Kokkos/CUDA translation unit through this include.
 */

namespace DiFfRG
{
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

      template <typename VectorType> struct _BlockVectorType;

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

      template <typename NT> struct _BlockVectorType<dealii::Vector<NT>> {
        using value = dealii::BlockVector<NT>;
      };
      template <typename NT> struct _BlockVectorType<dealii::BlockVector<NT>> {
        using value = dealii::BlockVector<NT>;
      };

      //--------------------------------------------------
      // Specified type helpers for distributed (PETSc) vectors
      //--------------------------------------------------
      //
      // PETSc's vectors are not templated on the number type -- PetscScalar is fixed
      // at PETSc configure time -- so these are full specializations and the number
      // type is read off the vector itself.
      //
      // Note what is deliberately NOT specialized here:
      //
      //  * _InverseSparseMatrixType. It exists only to give the *explicit* steppers
      //    (explicit_euler, boost_rk, boost_abm) a mass-matrix inverse, and those are
      //    not instantiated for PETSc -- they are not viable for these stiff flows and
      //    they route through common/eigen.hh, which assumes contiguous serial storage.
      //    Leaving it unspecialized turns an accidental instantiation into a clear
      //    "incomplete type" error instead of silently selecting a serial UMFPACK.
      //
      //  * A preconditioner trait for GMRES.hh's `PreconditionJacobi<SparseMatrixType>`
      //    default, which does not exist for PETSc matrices. None is needed: a default
      //    template argument is only instantiated when used, and the PETSc timestepper
      //    instantiations use PETScKrylov/PETScDirect, never GMRES or ScaledGMRES.
#ifdef DEAL_II_WITH_PETSC
      template <> struct _NumberType<dealii::PETScWrappers::MPI::Vector> {
        using value = dealii::PETScWrappers::MPI::Vector::value_type;
      };
      template <> struct _NumberType<dealii::PETScWrappers::MPI::BlockVector> {
        using value = dealii::PETScWrappers::MPI::BlockVector::value_type;
      };

      // A PETSc matrix owns its sparsity internally and is reinit'ed from a
      // DynamicSparsityPattern plus IndexSets (petsc_sparse_matrix.h:510), so there is
      // no persistent SparsityPattern object to hand back. Assemblers therefore keep
      // the DynamicSparsityPattern they build alive instead of discarding it after
      // copy_from().
      template <> struct _SparsityPattern<dealii::PETScWrappers::MPI::SparseMatrix> {
        using value = dealii::DynamicSparsityPattern;
      };
      template <> struct _SparsityPattern<dealii::PETScWrappers::MPI::BlockSparseMatrix> {
        using value = dealii::BlockDynamicSparsityPattern;
      };

      template <> struct _BlockVectorType<dealii::PETScWrappers::MPI::Vector> {
        using value = dealii::PETScWrappers::MPI::BlockVector;
      };
      template <> struct _BlockVectorType<dealii::PETScWrappers::MPI::BlockVector> {
        using value = dealii::PETScWrappers::MPI::BlockVector;
      };
#endif
    } // namespace internal

    template <typename VectorType> using NumberType = typename internal::_NumberType<VectorType>::value;

    template <typename SparseMatrixType>
    using SparsityPattern = typename internal::_SparsityPattern<SparseMatrixType>::value;

    template <typename SparseMatrixType>
    using InverseSparseMatrixType = typename internal::_InverseSparseMatrixType<SparseMatrixType>::value;

    /**
     * @brief The block-vector type belonging to a given vector type.
     *
     * Timesteppers need this for the FE+variables run path, where the state is one
     * block vector (block 0 = FE dofs, block 1 = variables). It used to be hard-wired
     * to dealii::BlockVector, which silently excluded every distributed vector type.
     */
    template <typename VectorType> using BlockVectorType = typename internal::_BlockVectorType<VectorType>::value;
  } // namespace get_type

  /**
   * @brief A vector type DiFfRG's timesteppers and assemblers can work with.
   *
   * Satisfied exactly when the get_type trait table knows the vector's number type,
   * i.e. for dealii::Vector/BlockVector and -- in an MPI build -- the PETSc MPI
   * vectors. This replaces a hard `static_assert(is_same_v<VectorType, Vector<...>>)`
   * that predated any distributed vector support.
   */
  template <typename VectorType>
  concept SupportedVectorType = requires { typename get_type::internal::_NumberType<VectorType>::value; };

  /**
   * @brief Whether a linear algebra type distributes its rows across MPI ranks.
   *
   * Everything the assemblers do differently under distribution keys off this one predicate, so
   * that the assembler bodies read identically for both policies and the branching lives in
   * la_policy.hh.
   */
  template <typename T> inline constexpr bool is_distributed_la = false;

#ifdef DEAL_II_WITH_PETSC
  template <> inline constexpr bool is_distributed_la<dealii::PETScWrappers::MPI::Vector> = true;
  template <> inline constexpr bool is_distributed_la<dealii::PETScWrappers::MPI::BlockVector> = true;
  template <> inline constexpr bool is_distributed_la<dealii::PETScWrappers::MPI::SparseMatrix> = true;
  template <> inline constexpr bool is_distributed_la<dealii::PETScWrappers::MPI::BlockSparseMatrix> = true;
#endif

  // ##############################################################################
  // Build-configuration defaults
  // ##############################################################################
  //
  // An MPI build defaults to distributed linear algebra on a partitioned mesh; a
  // serial build defaults to what it always did. Both halves of that choice -- the
  // triangulation and the vector/matrix pair -- must flip together, because a
  // distributed vector partitions its rows by rank ownership and that only exists
  // once the mesh has been partitioned.
  //
  // The condition is spelled once, here, and reused by rectangular_mesh.hh. Do not
  // re-derive it from DEAL_II_WITH_MPI alone: PETSc supplies the distributed vectors,
  // so an MPI-without-PETSc build must stay fully serial. Gating the mesh on MPI and
  // the vectors on PETSc would give that build a partitioned mesh with serial vectors,
  // which every static_assert accepts and which then assembles each rank's cells into
  // a full-size vector that is never summed.

#if defined(DEAL_II_WITH_MPI) && defined(DEAL_II_WITH_PETSC)
#define DIFFRG_DEFAULT_LA_DISTRIBUTED 1
#endif

#ifdef DIFFRG_DEFAULT_LA_DISTRIBUTED
  template <int dim> using DefaultTriangulation = dealii::parallel::shared::Triangulation<dim>;
#else
  template <int dim> using DefaultTriangulation = dealii::Triangulation<dim>;
#endif

  // ##############################################################################
  // The linear algebra that goes with a mesh
  // ##############################################################################
  //
  // A Discretization takes both a mesh and a vector/matrix pair, and the two must
  // agree: a distributed vector partitions its rows by rank ownership, which only
  // exists once the mesh is partitioned. Both directions of disagreement are wrong,
  // and the Discretizations static_assert against both.
  //
  // So derive the pair from the mesh rather than from the build configuration. That
  // makes the mesh the single dial: naming RectangularMeshSerial<dim> pins serial
  // linear algebra with it, which is what a test compared against a stored reference
  // needs in an MPI build tree. Defaulting the pair to the *build* configuration
  // instead made that pin silently insufficient -- the mesh went serial while the
  // vectors stayed PETSc, and the mismatch only surfaced as a static_assert naming
  // the mesh, not the vectors the caller never mentioned.
  //
  // A plain RectangularMesh<dim> still follows the build configuration, because its
  // own default triangulation does; nothing about the defaulted spelling changes.

#ifdef DIFFRG_DEFAULT_LA_DISTRIBUTED
  template <typename Mesh, typename NumberType>
  using LAVectorFor =
      std::conditional_t<Mesh::is_parallel, dealii::PETScWrappers::MPI::Vector, dealii::Vector<NumberType>>;
  template <typename Mesh, typename NumberType>
  using LASparseMatrixFor =
      std::conditional_t<Mesh::is_parallel, dealii::PETScWrappers::MPI::SparseMatrix, dealii::SparseMatrix<NumberType>>;
#else
  // Without PETSc there is no distributed vector to select, so the pair is serial
  // whatever the mesh says. A partitioned mesh is still nameable in an MPI-without-PETSc
  // build; the Discretization's static_assert rejects that combination rather than
  // letting it assemble per-rank cells into a full-size vector nothing ever sums.
  template <typename Mesh, typename NumberType> using LAVectorFor = dealii::Vector<NumberType>;
  template <typename Mesh, typename NumberType> using LASparseMatrixFor = dealii::SparseMatrix<NumberType>;
#endif
} // namespace DiFfRG
