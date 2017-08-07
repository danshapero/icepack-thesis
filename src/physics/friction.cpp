
#include <icepack/physics/friction.hpp>
#include <icepack/assembly.hpp>

using dealii::Tensor;
using dealii::SymmetricTensor;

namespace icepack
{
  BasalStress::BasalStress(const double m_) :
    m(m_),
    C([](const double beta){ return beta; }),
    dC([](const double){ return 1.0; })
  {}


  BasalStress::BasalStress(
    const double m_,
    const FrictionCoefficient C_,
    const FrictionCoefficient dC_
  ) : m(m_), C(C_), dC(dC_)
  {}


  dealii::Tensor<1, 2> BasalStress::operator()(
    const double beta,
    const dealii::Tensor<1, 2> u
  ) const
  {
    return C(beta) * std::pow(u.norm(), 1/m - 1) * u;
  }


  dealii::Tensor<1, 2> BasalStress::dbeta(
    const double beta,
    const dealii::Tensor<1, 2> u
  ) const
  {
    return dC(beta) * std::pow(u.norm(), 1/m - 1) * u;
  }


  namespace
  {
    const SymmetricTensor<2, 2> I = dealii::unit_symmetric_tensor<2>();
  }


  dealii::SymmetricTensor<2, 2> BasalStress::du(
    const double beta,
    const dealii::Tensor<1, 2> u
  ) const
  {
    const double c = C(beta) * std::pow(u.norm(), 1/m - 1);
    const Tensor<1, 2> gamma = u / u.norm();
    return c * (I + (1 - m) / m * outer_product(gamma, gamma));
  }


  Friction::Friction(const BasalStress& basal_stress_) :
    basal_stress(basal_stress_)
  {}


  double Friction::action(
    const Field<2>& beta,
    const VectorField<2>& u
  ) const
  {
    const auto& discretization = get_discretization(beta, u);
    double P = 0.0;

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(beta),
      evaluate::function(u)
    );

    const double m = basal_stress.m;

    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const double Beta = std::get<0>(values);
        const Tensor<1, 2> U = std::get<1>(values);
        const Tensor<1, 2> tau = basal_stress(Beta, U);
        P += m / (m + 1) * tau * U * dx;
      }
    }

    return P;
  }


  DualVectorField<2> Friction::derivative(
    const Field<2>& beta,
    const VectorField<2>& u,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    const auto& discretization = get_discretization(beta, u);
    DualVectorField<2> f(discretization);

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(beta),
      evaluate::function(u)
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
        const double Beta = std::get<0>(values);
        const Tensor<1, 2> U = std::get<1>(values);
        const Tensor<1, 2> tau = basal_stress(Beta, U);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, 2> phi =
            assembly_data.fe_values_view<1>().value(i, q);
          cell_derivative(i) += tau * phi * dx;
        }
      }

      std::get<1>(cell)->get_dof_indices(local_dof_ids);
      constraints.distribute_local_to_global(
        cell_derivative, local_dof_ids, f.coefficients());
    }

    return f;
  }


  double Friction::derivative(
    const Field<2>& beta,
    const VectorField<2>& u,
    const VectorField<2>& v
  ) const
  {
    const auto& discretization = get_discretization(beta, u, v);
    double df = 0.0;

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(beta),
      evaluate::function(u),
      evaluate::function(v)
    );

    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const double Beta = std::get<0>(values);
        const Tensor<1, 2> U = std::get<1>(values);
        const Tensor<1, 2> V = std::get<2>(values);
        const Tensor<1, 2> tau = basal_stress(Beta, U);

        df += tau * V * dx;
      }
    }

    return df;
  }


  dealii::SparseMatrix<double> Friction::hessian(
    const Field<2>& beta,
    const VectorField<2>& u,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    const auto& discretization = get_discretization(beta, u);
    dealii::SparseMatrix<double> A(discretization.vector().sparsity_pattern());

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(beta),
      evaluate::function(u)
    );

    const size_t dofs_per_cell =
      discretization(1).finite_element().dofs_per_cell;
    dealii::FullMatrix<double> cell_hessian(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);
    const auto& view = assembly_data.fe_values_view<1>();

    for (const auto& cell: discretization)
    {
      cell_hessian = 0;
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const double Beta = std::get<0>(values);
        const Tensor<1, 2> U = std::get<1>(values);
        const SymmetricTensor<2, 2> K = basal_stress.du(Beta, U);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, 2> phi_i = view.value(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            const Tensor<1, 2> phi_j = view.value(j, q);
            cell_hessian(i, j) += (phi_i * K * phi_j) * dx;
          }
        }
      }

      std::get<1>(cell)->get_dof_indices(local_dof_ids);
      constraints.distribute_local_to_global(cell_hessian, local_dof_ids, A);
    }

    return A;
  }

}

