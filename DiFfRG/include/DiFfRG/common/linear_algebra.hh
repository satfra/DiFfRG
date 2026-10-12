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
 * @brief Which vector, matrix and mesh types belong together.
 *
 * Two families of linear algebra are supported:
 * - serial: `dealii::Vector<double>` with `dealii::SparseMatrix<double>` (or the block versions), on a
 *   `dealii::Triangulation`;
 * - distributed (MPI builds with PETSc): `dealii::PETScWrappers::MPI::Vector` with
 *   `PETScWrappers::MPI::SparseMatrix`, on a `dealii::parallel::shared::Triangulation`, rows split across ranks.
 *
 * The get_type:: aliases map a vector or matrix type to its companions. Which family a discretization uses follows
 * from its mesh (LAVectorFor, LASparseMatrixFor); an unqualified mesh follows the build (DefaultTriangulation).
 * The linear solver that matches is chosen in timestepping/default_linear_solver.hh.
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

    /// The entry type of a vector, e.g. `double` for `dealii::Vector<double>`.
    template <typename VectorType> using NumberType = typename internal::_NumberType<VectorType>::value;

    /// The sparsity-pattern type a matrix is built from (`DynamicSparsityPattern` for PETSc matrices).
    template <typename SparseMatrixType>
    using SparsityPattern = typename internal::_SparsityPattern<SparseMatrixType>::value;

    /// A direct solver for a serial matrix (UMFPACK), used by the explicit steppers to invert the mass matrix.
    template <typename SparseMatrixType>
    using InverseSparseMatrixType = typename internal::_InverseSparseMatrixType<SparseMatrixType>::value;

    /**
     * @brief The block-vector type belonging to a given vector type.
     *
     * Timesteppers use it for the combined state of FE functions and variables: block 0 holds the FE degrees of
     * freedom, block 1 the variables.
     */
    template <typename VectorType> using BlockVectorType = typename internal::_BlockVectorType<VectorType>::value;
  } // namespace get_type

  /**
   * @brief A vector type DiFfRG's timesteppers and assemblers can work with.
   *
   * `dealii::Vector`, `dealii::BlockVector` and, in a PETSc build, the PETSc MPI vectors.
   */
  template <typename VectorType>
  concept SupportedVectorType = requires { typename get_type::internal::_NumberType<VectorType>::value; };

  /**
   * @brief Whether a linear algebra type distributes its rows across MPI ranks.
   *
   * True for the PETSc MPI vectors and matrices, false for the serial deal.II types.
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
  // serial build defaults to serial linear algebra. Both halves of that choice -- the
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
/// Defined in builds with MPI and PETSc, where meshes and linear algebra are distributed by default.
#define DIFFRG_DEFAULT_LA_DISTRIBUTED 1
#endif

#ifdef DIFFRG_DEFAULT_LA_DISTRIBUTED
  /**
   * @brief The triangulation an unqualified mesh (e.g. `RectangularMesh<dim>`) uses.
   *
   * `dealii::parallel::shared::Triangulation` in builds with MPI and PETSc, `dealii::Triangulation` otherwise.
   */
  template <int dim> using DefaultTriangulation = dealii::parallel::shared::Triangulation<dim>;
#else
  /**
   * @brief The triangulation an unqualified mesh (e.g. `RectangularMesh<dim>`) uses.
   *
   * `dealii::parallel::shared::Triangulation` in builds with MPI and PETSc, `dealii::Triangulation` otherwise.
   */
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
  // So the pair is derived from the mesh, not from the build configuration: naming
  // RectangularMeshSerial<dim> pins serial linear algebra even in an MPI build, while a
  // plain RectangularMesh<dim> follows the build configuration.

#ifdef DIFFRG_DEFAULT_LA_DISTRIBUTED
  /**
   * @brief The vector type for a discretization on `Mesh`: the PETSc MPI vector on a partitioned mesh
   * (`Mesh::is_parallel`), `dealii::Vector<NumberType>` otherwise.
   *
   * The PETSc vector's entry type is fixed by PETSc (`PetscScalar`), independent of `NumberType`. In builds without
   * PETSc this is always `dealii::Vector<NumberType>`. Use e.g. `RectangularMeshSerial<dim>` to force serial
   * linear algebra in an MPI build.
   */
  template <typename Mesh, typename NumberType>
  using LAVectorFor =
      std::conditional_t<Mesh::is_parallel, dealii::PETScWrappers::MPI::Vector, dealii::Vector<NumberType>>;
  /// The sparse-matrix type for a discretization on `Mesh`, matching LAVectorFor.
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
