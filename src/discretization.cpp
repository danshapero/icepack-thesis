
#include <icepack/discretization.hpp>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace icepack
{
  namespace
  {
    template <int dim>
    std::unique_ptr<dealii::FiniteElement<dim>>
    make_fe(const unsigned int p, const unsigned int rank)
    {
      if (rank == 0)
        return std::make_unique<dealii::FE_Q<dim>>(p);

      if (rank == 1)
        return
          std::make_unique<dealii::FESystem<dim>>(dealii::FE_Q<dim>(p), dim);

      throw;
    }
  }


  template <int dim>
  Discretization<dim>::Rank::Rank(
    const Triangulation<dim>& tria,
    const unsigned int p,
    const unsigned int rank
  ) : fe_(make_fe<dim>(p, rank)), dof_handler_(tria)
  {
    dof_handler_.distribute_dofs(*fe_);

    constraints_.clear();
    dealii::DoFTools::
      make_hanging_node_constraints(dof_handler_, constraints_);
    constraints_.close();

    dealii::DynamicSparsityPattern dsp(dof_handler_.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler_, dsp);
    sparsity_pattern_.copy_from(dsp);

    mass_matrix_.reinit(sparsity_pattern_);
    dealii::MatrixCreator::
      create_mass_matrix(dof_handler_, dealii::QGauss<dim>(p + 1), mass_matrix_);
  }


  template <int dim>
  Discretization<dim>::Rank::~Rank()
  {
    dof_handler_.clear();
  }

  template <int dim>
  const dealii::FiniteElement<dim>&
  Discretization<dim>::Rank::finite_element() const
  {
    return *fe_;
  }

  template <int dim>
  const dealii::DoFHandler<dim>&
  Discretization<dim>::Rank::dof_handler() const
  {
    return dof_handler_;
  }


  template <int dim>
  const dealii::ConstraintMatrix&
  Discretization<dim>::Rank::constraints() const
  {
    return constraints_;
  }


  template <int dim>
  const dealii::SparsityPattern&
  Discretization<dim>::Rank::sparsity_pattern() const
  {
    return sparsity_pattern_;
  }


  template <int dim>
  const dealii::SparseMatrix<double>&
  Discretization<dim>::Rank::mass_matrix() const
  {
    return mass_matrix_;
  }


  template <int dim>
  Discretization<dim>::Discretization(
    const Triangulation<dim>& tria,
    const unsigned int p
  ) : ranks_{{ {tria, p, 0}, {tria, p, 1} }}
  {}


  template <int dim>
  const Triangulation<dim>& Discretization<dim>::triangulation() const
  {
    return scalar().dof_handler().get_triangulation();
  }

  template <int dim>
  typename Discretization<dim>::iterator Discretization<dim>::begin() const
  {
    auto its = std::make_tuple(scalar().dof_handler().begin_active(),
                               vector().dof_handler().begin_active());
    return iterator(its);
  }

  template <int dim>
  typename Discretization<dim>::iterator Discretization<dim>:: end() const
  {
    auto its = std::make_tuple(scalar().dof_handler().end(),
                               vector().dof_handler().end());
    return iterator(its);
  }

  template <int dim>
  const typename Discretization<dim>::Rank& Discretization<dim>::scalar() const
  {
    return ranks_[0];
  }

  template <int dim>
  const typename Discretization<dim>::Rank& Discretization<dim>::vector() const
  {
    return ranks_[1];
  }

  template <int dim>
  const typename Discretization<dim>::Rank&
  Discretization<dim>::operator()(const unsigned int rank_) const
  {
    return ranks_[rank_];
  }

  template <int dim>
  dealii::QGauss<dim> Discretization<dim>::quad() const
  {
    const unsigned int p = scalar().finite_element().tensor_degree() + 1;
    return dealii::QGauss<dim>(p + 1);
  }


  template class Discretization<2>;
}
