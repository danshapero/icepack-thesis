
#include <icepack/inverse/error_functionals.hpp>
#include <icepack/assembly.hpp>

using dealii::Tensor;

namespace icepack
{
  template <int dim>
  double MeanSquareError<dim>::action(
    const VectorField<dim>& u,
    const VectorField<dim>& v,
    const Field<dim>& sigma
  ) const
  {
    const auto& discretization = get_discretization(u, v, sigma);
    double error = 0.0;

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<dim>(
      evaluate::function(u),
      evaluate::function(v),
      evaluate::function(sigma)
    );

    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const Tensor<1, dim> U = std::get<0>(values);
        const Tensor<1, dim> V = std::get<1>(values);
        const double Sigma = std::get<2>(values);

        error += 0.5 * (U - V).norm_square() / (Sigma * Sigma) * dx;
      }
    }

    return error;
  }


  template <int dim>
  DualVectorField<dim> MeanSquareError<dim>::derivative(
    const VectorField<dim>& u,
    const VectorField<dim>& v,
    const Field<dim>& sigma,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    const auto& discretization = get_discretization(u, v, sigma);
    DualVectorField<dim> f(discretization.shared_from_this());

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<dim>(
      evaluate::function(u),
      evaluate::function(v),
      evaluate::function(sigma)
    );

    const size_t dofs_per_cell =
      discretization(1).finite_element().dofs_per_cell;
    dealii::Vector<double> cell_derivative(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    for (const auto& cell: discretization)
    {
      cell_derivative = 0;
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const Tensor<1, dim> U = std::get<0>(values);
        const Tensor<1, dim> V = std::get<1>(values);
        const double Sigma = std::get<2>(values);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, dim> phi_i =
            assembly_data.template fe_values_view<1>().value(i, q);
          cell_derivative(i) += (U - V) * phi_i / (Sigma * Sigma) * dx;
        }
      }

      std::get<1>(cell)->get_dof_indices(local_dof_ids);
      constraints.distribute_local_to_global(
        cell_derivative, local_dof_ids, f.coefficients()
      );
    }

    return f;
  }

  template struct MeanSquareError<2>;
}

