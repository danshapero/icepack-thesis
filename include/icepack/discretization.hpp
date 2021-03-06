
#ifndef ICEPACK_DISCRETIZATION_HPP
#define ICEPACK_DISCRETIZATION_HPP

#include <memory>  // std::unique_ptr, std::shared_ptr
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/synchronous_iterator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_control.h>

namespace icepack
{
  template <int dim> class Discretization;

  template <int dim>
  std::shared_ptr<const Discretization<dim>> make_discretization(
    const dealii::Triangulation<dim>& tria,
    const unsigned int degree
  );


  /**
   * @brief Encapsulates all of the data needed to discretize scalar and vector
   * fields in some finite element basis.
   *
   * @ingroup field
   */
  template <int dim>
  class Discretization :
    public std::enable_shared_from_this<Discretization<dim>>
  {
  public:

    /**
     * @brief Helper class for data specific to a given tensor rank, i.e.
     * rank-0 for scalar and rank-1 for tensor fields.
     */
    class Rank
    {
    public:
      /**
       * Construct the ranked finite element data from the mesh `tria` and the
       * polynomial degree `p` of the finite element basis, i.e. `p = 1` for
       * bilinear finite elements, `p = 2` for biquadratic, etc.
       */
      Rank(
        const dealii::Triangulation<dim>& tria,
        const unsigned int p,
        const unsigned int rank
      );

      /**
       * Construct a new `Rank` object by transferring data from another `Rank`
       * object.
       */
      Rank(Rank&& rank) = default;

      /**
       * Free all of the heap memory that the `Rank` object stores.
       */
      ~Rank();


      /**
       * Return an immutable reference to the finite element object, which
       * describes how basis functions are constructed.
       */
      const dealii::FiniteElement<dim>& finite_element() const;

      /**
       * Return a `DoFHandler`, an object describing how geometric entities in
       * the mesh (vertices, lines, quads, hexes) are mapped to global degrees
       * of freedom in the finite element representation.
       */
      const dealii::DoFHandler<dim>& dof_handler() const;

      /**
       * Return the hanging node constraints on the degrees of freedom, i.e.
       * the constraints necessary to keep fields continuous across regions
       * where the mesh has been adaptively refined.
       */
      const dealii::ConstraintMatrix& constraints() const;

      /**
       * Return a sparsity pattern used to create matrices that represent
       * linear operators acting on fields.
       */
      const dealii::SparsityPattern& sparsity_pattern() const;

      /**
       * Return the mass matrix, i.e. the matrix that defines the inner product
       * between fields in terms of their Galerkin expansion coefficients.
       */
      const dealii::SparseMatrix<double>& mass_matrix() const;

      /**
       * Create a map describing the boundary values for homogeneous Dirichlet
       * boundary conditions.
       */
      std::map<dealii::types::global_dof_index, double>
      make_zero_boundary_values(const unsigned int boundary_id) const;

      /**
       * Create a set of constraints incorporating both handing nodes and
       * Dirichlet boundary conditions
       */
      dealii::ConstraintMatrix make_constraints(
        const std::set<dealii::types::boundary_id>& boundary_ids
      ) const;

    protected:
      std::unique_ptr<dealii::FiniteElement<dim>> fe_;
      dealii::DoFHandler<dim> dof_handler_;
      dealii::ConstraintMatrix constraints_;
      dealii::SparsityPattern sparsity_pattern_;
      dealii::SparseMatrix<double> mass_matrix_;
    };


    /**
     * Destructor; frees all memory used by the discretization.
     */
    virtual ~Discretization();


    /**
     * Return a reference to the underlying geometry for this discretization.
     */
    const dealii::Triangulation<dim>& triangulation() const;

    /**
     * Return a quadrature object sufficient to exactly integrate polynomials of
     * degree `2 * p + 1`, where `p` is the degree of the finite element basis
     * for this discretization.
     */
    dealii::QGauss<dim> quad() const;

    /**
     * Same as above, but for quadrature over faces of a mesh rather than cells.
     */
    dealii::QGauss<dim - 1> face_quad() const;


    /**
     * Return a reference to the `Rank` object for scalar fields.
     */
    const Rank& scalar() const;

    /**
     * Return a reference to the `Rank` object for vector fields.
     */
    const Rank& vector() const;

    /**
     * Return a reference to a `Rank` object of either tensor rank.
     */
    const Rank& operator()(unsigned int rank_) const;


    /**
     * Local typedef for iterators used to traverse over all of the degrees of
     * freedom of some finite element field.
     */
    using dof_iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

    /**
     * This type stores a pair of DoF iterators, one for scalar fields and one
     * for vector fields. Rather construct both iterators separately and have
     * to manually write the code for stepping through each in sync, you can
     * use this single iterator to evaluate quantities that depend on both
     * scalar and vector fields.
     */
    using iterator =
      dealii::SynchronousIterators<std::tuple<dof_iterator, dof_iterator>>;

    /**
     * Return an iterator pointing to the start of the scalar and vector
     * degrees of freedom.
     */
    iterator begin() const;

    /**
     * Return an iterator point to the end of the scalar and vector degrees of
     * freedom.
     */
    iterator end() const;

  protected:
    /**
     * Construct a discretization from the geometry and the desired degree of
     * the finite element basis, i.e. `degree = 1` for bilinear elements, 2 for
     * biquadratic elements, etc.
     *
     * This constructor is not public because discretization objects should be
     * created only through a shared pointer.
     */
    Discretization(
      const dealii::Triangulation<dim>& tria,
      const unsigned int degree
    );

    // TODO: Add an overload of the constructor for r-value references to the
    // mesh so that it can be moved rather than copied.

    /**
     * Give the function `make_discretization` access to the constructor for
     * this class.
     */
    friend std::shared_ptr<const Discretization<dim>> make_discretization<dim>(
      const dealii::Triangulation<dim>&,
      const unsigned int
    );

    const dealii::Triangulation<dim> tria_;
    std::array<Rank, 2> ranks_;
  };


  /**
   * @brief Make a discretization object used to keep track of the degrees of
   * freedom and the finite element for fields defined over a given domain.
   *
   * This function will copy the mesh into the discretization, which always
   * owns the underlying geometry. Changes made to the original mesh passed to
   * this function, i.e. refinement or smoothing, will not be reflected in the
   * discretization.
   *
   * Rather than create a `Discretization` object itself, this function makes a
   * smart pointer that allows it to be shared among several objects. In our
   * case, those objects are the many fields and vector fields that will be
   * created using this discretization. The smart pointer will clean up the
   * discretization once there are no more objects that refer to it.
   *
   * @ingroup field
   */
  template <int dim>
  std::shared_ptr<const Discretization<dim>> make_discretization(
    const dealii::Triangulation<dim>& tria,
    const unsigned int degree
  )
  {
    return std::shared_ptr<const Discretization<dim>>
      (new Discretization<dim>(tria, degree));
  }

} // namespace icepack

#endif

