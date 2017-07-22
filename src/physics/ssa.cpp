
#include <icepack/physics/ssa.hpp>
#include <icepack/physics/constants.hpp>
#include <icepack/numerics/optimization.hpp>
#include <icepack/assembly.hpp>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/sparse_direct.h>

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
    membrane_stress(rheology_)
  {}


  double Viscosity::action(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u
  ) const
  {
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::function(theta),
      evaluate::symmetric_gradient(u)
    );

    const double n = membrane_stress.rheology.n;
    const auto Functional =
    [&](const double H, const double T, const SymmetricTensor<2, 2> eps)
    {
      const SymmetricTensor<2, 2> M = membrane_stress(T, eps);
      return n / (n + 1) * H * (M * eps);
    };

    return integrate(Functional, assembly_data);
  }


  DualVectorField<2> Viscosity::derivative(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::function(theta),
      evaluate::symmetric_gradient(u)
    );

    const auto Functional =
    [&](
      const double H,
      const double T,
      const SymmetricTensor<2, 2> eps_u,
      const SymmetricTensor<2, 2> eps_v
    )
    {
      const SymmetricTensor<2, 2> M = membrane_stress(T, eps_u);
      return H * (M * eps_v);
    };

    const auto eps_v = vector_shape_fn<2>::symmetric_gradient();
    return integrate(Functional, constraints, eps_v, assembly_data);
  }


  double Viscosity::derivative(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u,
    const VectorField<2>& v
  ) const
  {
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::function(theta),
      evaluate::symmetric_gradient(u),
      evaluate::symmetric_gradient(v)
    );

    const auto Functional =
    [&](
      const double H,
      const double T,
      const SymmetricTensor<2, 2> eps_u,
      const SymmetricTensor<2, 2> eps_v
    )
    {
      const SymmetricTensor<2, 2> M = membrane_stress(T, eps_u);
      return H * (M * eps_v);
    };

    return integrate(Functional, assembly_data);
  }


  dealii::SparseMatrix<double> Viscosity::hessian(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::function(theta),
      evaluate::symmetric_gradient(u)
    );

    const auto Functional =
    [&](
      const double H,
      const double T,
      const SymmetricTensor<2, 2> eps_u,
      const SymmetricTensor<2, 2> eps_v,
      const SymmetricTensor<2, 2> eps_w
    )
    {
      const SymmetricTensor<4, 2> K = membrane_stress.du(T, eps_u);
      return H * (eps_v * K * eps_w);
    };

    const auto eps_v = vector_shape_fn<2>::symmetric_gradient();
    const auto eps_w = vector_shape_fn<2>::symmetric_gradient();
    return integrate(Functional, constraints, eps_v, eps_w, assembly_data);
  }


  Gravity::Gravity()
  {}

  double Gravity::action(const Field<2>& h, const VectorField<2>& u) const
  {
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::divergence(u)
    );

    using namespace icepack::constants;
    const double Rho = rho_ice * (1 - rho_ice / rho_water);
    const auto Functional = [&](const double H, const double div_U)
    {
      return -0.5 * Rho * gravity * H * H * div_U;
    };

    return integrate(Functional, assembly_data);
  }


  DualVectorField<2> Gravity::derivative(
    const Field<2>& h,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    auto assembly_data = make_assembly_data<2>(evaluate::function(h));

    using namespace icepack::constants;
    const double Rho = rho_ice * (1 - rho_ice / rho_water);
    const auto Functional = [&](const double H, const double div_v)
    {
      return -0.5 * Rho * gravity * H * H * div_v;
    };

    const auto div_v = vector_shape_fn<2>::divergence();
    return integrate(Functional, constraints, div_v, assembly_data);
  }


  double Gravity::derivative(const Field<2>& h, const VectorField<2>& v) const
  {
    // The gravitational stress action is linear in the velocity, so the
    // directional derivative is just itself!
    return action(h, v);
  }


  IceShelf::IceShelf(
    const std::set<dealii::types::boundary_id>& dirichlet_boundary_ids_,
    const Viscosity& viscosity_,
    const double tolerance_
  ) : dirichlet_boundary_ids(dirichlet_boundary_ids_),
      gravity(),
      viscosity(viscosity_),
      tolerance(tolerance_)
  {}


  namespace
  {
    dealii::ConstraintMatrix make_constraints(
      const typename Discretization<2>::Rank& rank,
      const std::set<dealii::types::boundary_id>& boundary_ids
    )
    {
      const auto& dof_handler = rank.dof_handler();
      const auto merge_behavior = dealii::ConstraintMatrix::right_object_wins;

      dealii::ConstraintMatrix constraints;
      for (const auto& id: boundary_ids)
      {
        dealii::ConstraintMatrix boundary_constraints;
        dealii::DoFTools::make_zero_boundary_constraints(
          dof_handler, id, boundary_constraints);
        constraints.merge(boundary_constraints, merge_behavior);
      }

      const auto& hanging_node_constraints = rank.constraints();
      constraints.merge(hanging_node_constraints, merge_behavior);

      constraints.close();
      return constraints;
    }
  }


  VectorField<2> IceShelf::solve(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u0,
    const SolveOptions options
  ) const
  {
    const auto& discretization = get_discretization(h, theta, u0);
    const dealii::ConstraintMatrix constraints =
      make_constraints(discretization.vector(), dirichlet_boundary_ids);

    const auto F =
      [&](const VectorField<2>& v) -> double
      {
        return gravity.action(h, v) + viscosity.action(h, theta, v);
      };

    const auto dF =
      [&](const VectorField<2>& v, const VectorField<2>& q) -> double
      {
        return gravity.derivative(h, q) + viscosity.derivative(h, theta, v, q);
      };

    const auto P =
      [&](const VectorField<2>& v) -> VectorField<2>
      {
        VectorField<2> p(discretization);

        // Compute the derivative of the action.
        DualVectorField<2> df =
          gravity.derivative(h, constraints) +
          viscosity.derivative(h, theta, v, constraints);

        // Compute the second derivative of the action.
        const dealii::SparseMatrix<double> A =
          viscosity.hessian(h, theta, v, constraints);

        // Solve for the search direction using Newton's method.
        dealii::SparseDirectUMFPACK G;
        G.initialize(A);
        G.vmult(p.coefficients(), df.coefficients());
        constraints.distribute(p.coefficients());
        p *= -1;

        return p;
      };

    const double initial_viscous_action = viscosity.action(h, theta, u0);
    const double tol = initial_viscous_action * tolerance;
    return numerics::newton_search(u0, F, dF, P, tol, options);
  }

} // namespace icepack

