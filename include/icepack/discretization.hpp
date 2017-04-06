
#ifndef ICEPACK_DISCRETIZATION_HPP
#define ICEPACK_DISCRETIZATION_HPP

#include <memory>  // std::unique_ptr
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/synchronous_iterator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_control.h>

namespace icepack
{
  using dealii::Triangulation;
  using dealii::QGauss;
  using dealii::FiniteElement;
  using dealii::DoFHandler;
  using dealii::SparsityPattern;
  using dealii::ConstraintMatrix;
  using dealii::SparseMatrix;
  using dealii::SolverControl;
  using dealii::SmartPointer;


  template <int dim>
  class Discretization : public dealii::Subscriptor
  {
  public:

    class Rank
    {
    public:
      Rank(const Triangulation<dim>& tria, unsigned int p, unsigned int rank);
      Rank(Rank&& rank) = default;
      ~Rank();

      const FiniteElement<dim>& finite_element() const;
      const DoFHandler<dim>& dof_handler() const;
      const ConstraintMatrix& constraints() const;
      const SparsityPattern& sparsity_pattern() const;
      const SparseMatrix<double>& mass_matrix() const;

    protected:
      std::unique_ptr<FiniteElement<dim>> fe_;
      DoFHandler<dim> dof_handler_;
      ConstraintMatrix constraints_;
      SparsityPattern sparsity_pattern_;
      SparseMatrix<double> mass_matrix_;
    };

    Discretization(const Triangulation<dim>& tria, unsigned int degree);

    const Triangulation<dim>& triangulation() const;
    const Rank& scalar() const;
    const Rank& vector() const;
    const Rank& operator()(unsigned int rank_) const;

    QGauss<dim> quad() const;

  protected:
    std::array<Rank, 2> ranks_;
  };

} // namespace icepack


#endif
