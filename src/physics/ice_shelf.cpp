
#include <icepack/physics/ice_shelf.hpp>
#include <icepack/physics/constants.hpp>
#include <icepack/numerics/optimization.hpp>
#include <icepack/assembly.hpp>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/sparse_direct.h>

namespace icepack
{
  using namespace constants;

  Gravity::Gravity()
  {}

  double Gravity::action(const Field<2>& h, const VectorField<2>& u) const
  {
    const auto& discretization = get_discretization(h, u);
    double P = 0.0;

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::divergence(u)
    );

    using namespace icepack::constants;
    const double Rho = rho_ice * (1 - rho_ice / rho_water);

    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const double H = std::get<0>(values);
        const double div_U = std::get<1>(values);
        P -= 0.5 * Rho * gravity * H * H * div_U * dx;
      }
    }

    return P;
  }


  DualVectorField<2> Gravity::derivative(
    const Field<2>& h,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    const auto& discretization = get_discretization(h);
    DualVectorField<2> tau(discretization);

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(evaluate::function(h));

    const size_t dofs_per_cell =
      discretization(1).finite_element().dofs_per_cell;
    dealii::Vector<double> cell_derivative(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    using namespace icepack::constants;
    const double Rho = rho_ice * (1 - rho_ice / rho_water);

    for (const auto& cell: discretization)
    {
      cell_derivative = 0;
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const double H = std::get<0>(assembly_data.values(q));
        const double Tau = -0.5 * Rho * gravity * H * H;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const double div_phi_i =
            assembly_data.fe_values_view<1>().divergence(i, q);
          cell_derivative(i) += Tau * div_phi_i * dx;
        }
      }

      std::get<1>(cell)->get_dof_indices(local_dof_ids);
      constraints.distribute_local_to_global(
        cell_derivative, local_dof_ids, tau.coefficients()
      );
    }

    return tau;
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

