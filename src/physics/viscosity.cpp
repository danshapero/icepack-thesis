
#include <icepack/physics/viscosity.hpp>
#include <icepack/physics/constants.hpp>
#include <icepack/assembly.hpp>

using dealii::SymmetricTensor;

namespace icepack
{
  using namespace constants;

  const double transition_temperature = 263.15;  // K
  const double A0_cold = 3.985e-13 * year * 1.0e18;  // MPa^{-3} yr^{-1}
  const double A0_warm = 1.916e3 * year * 1.0e18;
  const double Q_cold = 60; // kJ / mol
  const double Q_warm = 139;


  double rate_factor(const double temperature)
  {
    const bool cold = (temperature < transition_temperature);
    const double A0 = cold ? A0_cold : A0_warm;
    const double Q = cold ? Q_cold : Q_warm;
    const double r = Q / (ideal_gas * temperature);
    return A0 * std::exp(-r);
  }

  namespace
  {
    double drate_factor(const double temperature)
    {
      const bool cold = (temperature < transition_temperature);
      const bool A0 = cold ? A0_cold : A0_warm;
      const double Q = cold ? Q_cold : Q_warm;
      const double r = Q / (ideal_gas * temperature);
      return A0 * r / temperature * std::exp(-r);
    }
  }



  MembraneStress::MembraneStress(const double n_) :
    n(n_), A(rate_factor), dA(drate_factor)
  {}


  MembraneStress::
  MembraneStress(const double n_, const RateFactor A_, const RateFactor dA_) :
    n(n_), A(A_), dA(dA_)
  {}


  double MembraneStress::B(const double theta) const
  {
    return std::pow(A(theta), -1/n);
  }


  double MembraneStress::dB(const double theta) const
  {
    const double a = A(theta);
    const double da = dA(theta);
    return -1/n * std::pow(a, -1/n - 1) * da;
  }


  namespace
  {
    const SymmetricTensor<2, 2> I = dealii::unit_symmetric_tensor<2>();
    const SymmetricTensor<4, 2> II = dealii::identity_tensor<2>();
    const SymmetricTensor<4, 2> D = II + outer_product(I, I);
  }

  SymmetricTensor<2, 2> MembraneStress::operator()(
    const double theta,
    const SymmetricTensor<2, 2> eps
  ) const
  {
    const double tr = trace(eps);
    const double eps_e = std::sqrt((eps * eps + tr * tr) / 2);
    const double mu = 0.5 * B(theta) * std::pow(eps_e, 1/n - 1);
    return 2 * mu * (eps + tr * I);
  }

  SymmetricTensor<2, 2> MembraneStress::dtheta(
    const double theta,
    const SymmetricTensor<2, 2> eps
  ) const
  {
    const double tr = trace(eps);
    const double eps_e = std::sqrt((eps * eps + tr * tr) / 2);
    const double dmu = 0.5 * dB(theta) * std::pow(eps_e, 1/n - 1);
    return 2 * dmu * (eps + tr * I);
  }


  SymmetricTensor<4, 2> MembraneStress::du(
    const double theta,
    const SymmetricTensor<2, 2> eps
  ) const
  {
    const double tr = trace(eps);
    const double eps_e = std::sqrt((eps * eps + tr * tr) / 2);
    const double mu = 0.5 * B(theta) * std::pow(eps_e, 1/n - 1);
    const SymmetricTensor<2, 2> gamma = (eps + tr * I) / eps_e;
    return 2 * mu * (D + (1 - n)/(2*n) * outer_product(gamma, gamma));
  }


  Viscosity::Viscosity(const MembraneStress& membrane_stress_) :
    membrane_stress(membrane_stress_)
  {}


  double Viscosity::action(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u
  ) const
  {
    const auto& discretization = get_discretization(h, theta, u);
    double P = 0.0;

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::function(theta),
      evaluate::symmetric_gradient(u)
    );

    const double n = membrane_stress.n;

    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const double H = std::get<0>(values);
        const double T = std::get<1>(values);
        const SymmetricTensor<2, 2> eps = std::get<2>(values);
        const SymmetricTensor<2, 2> M = membrane_stress(T, eps);
        P += n / (n + 1) * H * (M * eps) * dx;
      }
    }

    return P;
  }


  DualVectorField<2> Viscosity::derivative(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    const auto& discretization = get_discretization(h, theta, u);
    DualVectorField<2> f(discretization.shared_from_this());

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::function(theta),
      evaluate::symmetric_gradient(u)
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
        const double H = std::get<0>(values);
        const double T = std::get<1>(values);
        const SymmetricTensor<2, 2> eps = std::get<2>(values);
        const SymmetricTensor<2, 2> M = membrane_stress(T, eps);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const SymmetricTensor<2, 2> eps_phi_i =
            assembly_data.fe_values_view<1>().symmetric_gradient(i, q);
          cell_derivative(i) += H * (M * eps_phi_i) * dx;
        }
      }

      std::get<1>(cell)->get_dof_indices(local_dof_ids);
      constraints.distribute_local_to_global(
        cell_derivative, local_dof_ids, f.coefficients()
      );
    }

    return f;
  }


  double Viscosity::derivative(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u,
    const VectorField<2>& v
  ) const
  {
    const auto& discretization = get_discretization(h, theta, u, v);
    double df = 0.0;

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::function(theta),
      evaluate::symmetric_gradient(u),
      evaluate::symmetric_gradient(v)
    );

    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const double H = std::get<0>(values);
        const double T = std::get<1>(values);
        const SymmetricTensor<2, 2> eps_u = std::get<2>(values);
        const SymmetricTensor<2, 2> eps_v = std::get<3>(values);
        const SymmetricTensor<2, 2> M = membrane_stress(T, eps_u);

        df += H * (M * eps_v) * dx;
      }
    }

    return df;
  }


  dealii::SparseMatrix<double> Viscosity::hessian(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    const auto& discretization = get_discretization(h, theta, u);
    dealii::SparseMatrix<double> A(discretization.vector().sparsity_pattern());

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::function(theta),
      evaluate::symmetric_gradient(u)
    );

    const size_t dofs_per_cell =
      discretization(1).finite_element().dofs_per_cell;
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);
    const auto& view = assembly_data.fe_values_view<1>();

    for (const auto& cell: discretization)
    {
      cell_matrix = 0;
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const double H = std::get<0>(values);
        const double T = std::get<1>(values);
        const SymmetricTensor<2, 2> eps = std::get<2>(values);
        const SymmetricTensor<4, 2> K = membrane_stress.du(T, eps);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const SymmetricTensor<2, 2> eps_i = view.symmetric_gradient(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            const SymmetricTensor<2, 2> eps_j = view.symmetric_gradient(j, q);
            cell_matrix(i, j) += H * (eps_i * K * eps_j) * dx;
          }
        }
      }

      std::get<1>(cell)->get_dof_indices(local_dof_ids);
      constraints.distribute_local_to_global(cell_matrix, local_dof_ids, A);
    }

    return A;
  }


  DualField<2> Viscosity::mixed_derivative(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u,
    const VectorField<2>& v,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    const auto& discretization = get_discretization(h, theta, u, v);
    DualField<2> f(discretization.shared_from_this());

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::function(theta),
      evaluate::symmetric_gradient(u),
      evaluate::symmetric_gradient(v)
    );

    const size_t dofs_per_cell =
      discretization(0).finite_element().dofs_per_cell;
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
        const double H = std::get<0>(values);
        const double Theta = std::get<1>(values);
        const SymmetricTensor<2, 2> eps_u = std::get<2>(values);
        const SymmetricTensor<2, 2> eps_v = std::get<3>(values);
        const SymmetricTensor<2, 2> dM = membrane_stress.dtheta(Theta, eps_u);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const double phi_i = assembly_data.fe_values_view<0>().value(i, q);
          cell_derivative(i) += H * (dM * eps_v) * phi_i * dx;
        }
      }

      std::get<0>(cell)->get_dof_indices(local_dof_ids);
      constraints.distribute_local_to_global(
        cell_derivative, local_dof_ids, f.coefficients()
      );
    }

    return f;
  }
}

