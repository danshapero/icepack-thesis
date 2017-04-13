
#include <icepack/physics/constants.hpp>
#include <icepack/physics/ssa.hpp>

using dealii::SymmetricTensor;
using dealii::FEValues;

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

  double drate_factor(const double temperature)
  {
    const bool cold = (temperature < transition_temperature);
    const bool A0 = cold ? A0_cold : A0_warm;
    const double Q = cold ? Q_cold : Q_warm;
    const double r = Q / (ideal_gas * temperature);
    return A0 * r / temperature * std::exp(-r);
  }


  ViscousRheology::ViscousRheology(const double n_) : n(n_)
  {}

  double ViscousRheology::operator()(const double theta) const
  {
    const double A = rate_factor(theta);
    return std::pow(A, -1/n);
  }

  double ViscousRheology::dtheta(const double theta) const
  {
    const double A = rate_factor(theta);
    const double dA = drate_factor(theta);
    return -1/n * std::pow(A, -1/n - 1) * dA;
  }


  MembraneStress::MembraneStress(const ViscousRheology& rheology_) :
    rheology(rheology_)
  {}


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
    const double n = rheology.n;
    const double B = rheology(theta);
    const double tr = trace(eps);
    const double eps_e = std::sqrt((eps * eps + tr * tr) / 2);
    const double mu = 0.5 * B * std::pow(eps_e, 1/n - 1);
    return 2 * mu * (eps + tr * I);
  }

  SymmetricTensor<2, 2> MembraneStress::dtheta(
    const double theta,
    const SymmetricTensor<2, 2> eps
  ) const
  {
    const double n = rheology.n;
    const double dB = rheology.dtheta(theta);
    const double tr = trace(eps);
    const double eps_e = std::sqrt((eps * eps + tr * tr) / 2);
    const double dmu = 0.5 * dB * std::pow(eps_e, 1/n - 1);
    return 2 * dmu * (eps + tr * I);
  }


  SymmetricTensor<4, 2> MembraneStress::du(
    const double theta,
    const SymmetricTensor<2, 2> eps
  ) const
  {
    const double B = rheology(theta);
    const double n = rheology.n;
    const double tr = trace(eps);
    const double eps_e = std::sqrt((eps * eps + tr * tr) / 2);
    const double mu = 0.5 * B * std::pow(eps_e, 1/n - 1);
    const SymmetricTensor<2, 2> gamma = (eps + tr * I) / eps_e;
    return 2 * mu * (D + (1 - n)/(2*n) * outer_product(gamma, gamma));
  }


  Viscosity::Viscosity(const ViscousRheology& rheology_) :
    rheology(rheology_), membrane_stress(rheology)
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
    const unsigned int n_q_points = quad.size();

    std::vector<double> h_values(n_q_points);
    std::vector<double> theta_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> eps_values(n_q_points);

    using icepack::DefaultFlags::flags;

    FEValues<2> u_fe_values(discretization(1).finite_element(), quad, flags);
    const auto& exv = u_fe_values[dealii::FEValuesExtractors::Vector(0)];

    FEValues<2> h_fe_values(discretization(0).finite_element(), quad, flags);
    const auto& exs = h_fe_values[dealii::FEValuesExtractors::Scalar(0)];

    const double n = rheology.n;

    for (const auto& it: discretization)
    {
      u_fe_values.reinit(std::get<1>(it));
      h_fe_values.reinit(std::get<0>(it));

      exs.get_function_values(h.coefficients(), h_values);
      exs.get_function_values(theta.coefficients(), theta_values);
      exv.get_function_symmetric_gradients(u.coefficients(), eps_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double dx = u_fe_values.JxW(q);
        const double H = h_values[q];
        const double T = theta_values[q];
        const SymmetricTensor<2, 2> eps = eps_values[q];
        const SymmetricTensor<2, 2> M = membrane_stress(T, eps);
        P += n / (n + 1) * H * (eps * M) * dx;
      }
    }

    return P;
  }


  DualVectorField<2> Viscosity::derivative(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u
  ) const
  {
    const auto& discretization = get_discretization(h, theta, u);
    DualVectorField<2> f(discretization);

    const auto quad = discretization.quad();
    const unsigned int n_q_points = quad.size();

    std::vector<double> h_values(n_q_points);
    std::vector<double> theta_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> eps_values(n_q_points);

    const auto& fe = discretization(1).finite_element();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    Vector<double> cell_derivative(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    using icepack::DefaultFlags::flags;

    FEValues<2> u_fe_values(fe, quad, flags);
    const auto& exv = u_fe_values[dealii::FEValuesExtractors::Vector(0)];

    FEValues<2> h_fe_values(discretization(0).finite_element(), quad, flags);
    const auto& exs = h_fe_values[dealii::FEValuesExtractors::Scalar(0)];

    for (const auto& it: discretization)
    {
      cell_derivative = 0;
      u_fe_values.reinit(std::get<1>(it));
      h_fe_values.reinit(std::get<0>(it));

      exs.get_function_values(h.coefficients(), h_values);
      exs.get_function_values(theta.coefficients(), theta_values);
      exv.get_function_symmetric_gradients(u.coefficients(), eps_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double dx = u_fe_values.JxW(q);
        const double H = h_values[q];
        const double T = theta_values[q];
        const SymmetricTensor<2, 2> eps = eps_values[q];
        const SymmetricTensor<2, 2> M = membrane_stress(T, eps);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const SymmetricTensor<2, 2> eps_phi_i = exv.symmetric_gradient(i, q);
          cell_derivative(i) += H * (eps_phi_i * M) * dx;
        }
      }

      std::get<1>(it)->get_dof_indices(local_dof_ids);
      f.coefficients().add(local_dof_ids, cell_derivative);
    } // end of loop over `it`

    return f;
  }


  dealii::SparseMatrix<double> Viscosity::hessian(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u
  ) const
  {
    const auto& discretization = get_discretization(h, theta, u);
    dealii::SparseMatrix<double> A(discretization.vector().sparsity_pattern());

    const auto quad = discretization.quad();
    const unsigned int n_q_points = quad.size();

    std::vector<double> h_values(n_q_points);
    std::vector<double> theta_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> eps_values(n_q_points);

    const auto& fe = discretization(1).finite_element();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    using icepack::DefaultFlags::flags;

    FEValues<2> u_fe_values(fe, quad, flags);
    const auto& exv = u_fe_values[dealii::FEValuesExtractors::Vector(0)];

    FEValues<2> h_fe_values(discretization(0).finite_element(), quad, flags);
    const auto& exs = h_fe_values[dealii::FEValuesExtractors::Scalar(0)];

    for (const auto& it: discretization)
    {
      cell_matrix = 0;
      u_fe_values.reinit(std::get<1>(it));
      h_fe_values.reinit(std::get<0>(it));

      exs.get_function_values(h.coefficients(), h_values);
      exs.get_function_values(theta.coefficients(), theta_values);
      exv.get_function_symmetric_gradients(u.coefficients(), eps_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double dx = u_fe_values.JxW(q);
        const double H = h_values[q];
        const double T = theta_values[q];
        const SymmetricTensor<2, 2> eps = eps_values[q];
        const SymmetricTensor<4, 2> K = membrane_stress.du(T, eps);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const SymmetricTensor<2, 2> eps_phi_i = exv.symmetric_gradient(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            const SymmetricTensor<2, 2> eps_phi_j = exv.symmetric_gradient(j, q);
            cell_matrix(i, j) += H * (eps_phi_i * K * eps_phi_j) * dx;
          }
        }
      }

      std::get<1>(it)->get_dof_indices(local_dof_ids);
      A.add(local_dof_ids, cell_matrix);
    }

    return A;
  }


  Gravity::Gravity()
  {}

  double Gravity::action(const Field<2>& h, const VectorField<2>& u) const
  {
    const auto& discretization = get_discretization(h, u);
    double P = 0.0;

    const auto quad = discretization.quad();
    const unsigned int n_q_points = quad.size();

    std::vector<double> h_values(n_q_points);
    std::vector<double> div_u_values(n_q_points);

    using icepack::DefaultFlags::flags;

    FEValues<2> u_fe_values(discretization(1).finite_element(), quad, flags);
    const auto& exv = u_fe_values[dealii::FEValuesExtractors::Vector(0)];

    FEValues<2> h_fe_values(discretization(0).finite_element(), quad, flags);
    const auto& exs = h_fe_values[dealii::FEValuesExtractors::Scalar(0)];

    using namespace icepack::constants;
    const double Rho = rho_ice * (1 - rho_ice / rho_water);

    for (const auto& it: discretization)
    {
      u_fe_values.reinit(std::get<1>(it));
      h_fe_values.reinit(std::get<0>(it));

      exs.get_function_values(h.coefficients(), h_values);
      exv.get_function_divergences(u.coefficients(), div_u_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double dx = h_fe_values.JxW(q);
        const double H = h_values[q];
        const double divU = div_u_values[q];
        P -= 0.5 * Rho * gravity * H * H * divU * dx;
      }
    }

    return P;
  }


  DualVectorField<2> Gravity::derivative(const Field<2>& h) const
  {
    const auto& discretization = get_discretization(h);
    DualVectorField<2> tau(discretization);

    const auto quad = discretization.quad();
    const unsigned int n_q_points = quad.size();

    std::vector<double> h_values(n_q_points);

    const auto& fe = discretization(1).finite_element();

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    Vector<double> cell_derivative(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    using namespace icepack::constants;
    const double Rho = rho_ice * (1 - rho_ice / rho_water);

    using DefaultFlags::flags;
    FEValues<2> tau_fe_values(fe, quad, flags);
    const auto& exv = tau_fe_values[dealii::FEValuesExtractors::Vector(0)];

    FEValues<2> h_fe_values(discretization(0).finite_element(), quad, flags);
    const auto& exs = h_fe_values[dealii::FEValuesExtractors::Scalar(0)];

    for (const auto& it: discretization)
    {
      cell_derivative = 0;
      tau_fe_values.reinit(std::get<1>(it));
      h_fe_values.reinit(std::get<0>(it));

      exs.get_function_values(h.coefficients(), h_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double dx = tau_fe_values.JxW(q);
        const double H = h_values[q];
        const double Tau = -0.5 * Rho * gravity * H * H;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const double div_phi_i = exv.divergence(i, q);
          cell_derivative(i) += Tau * div_phi_i * dx;
        }
      }

      std::get<1>(it)->get_dof_indices(local_dof_ids);
      tau.coefficients().add(local_dof_ids, cell_derivative);
    } // end of loop over `it`

    return tau;
  }


} // namespace icepack
