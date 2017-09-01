
#include <icepack/physics/ice_stream.hpp>
#include <icepack/physics/constants.hpp>
#include <icepack/assembly.hpp>
#include <deal.II/base/geometry_info.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/sparse_direct.h>

namespace icepack
{
  using dealii::Tensor;
  using dealii::SymmetricTensor;
  using namespace icepack::constants;

  constexpr unsigned int faces_per_cell =
    dealii::GeometryInfo<2>::faces_per_cell;

  Gravity::Gravity(const std::set<dealii::types::boundary_id>& ids) :
    dirichlet_ids(ids)
  {}


  double Gravity::action(
    const Field<2>& h,
    const Field<2>& s,
    const VectorField<2>& u
  ) const
  {
    const auto& discretization = get_discretization(h, s, u);
    double P = 0.0;

    const auto quad = discretization.quad();
    auto assembly_data = make_assembly_data<2>(
      evaluate::function(h),
      evaluate::gradient(s),
      evaluate::function(u)
    );

    const auto face_quad = discretization.face_quad();
    auto assembly_face_data = make_assembly_face_data<2>(
      evaluate::function(h),
      evaluate::function(s),
      evaluate::function(u)
    );

    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);
        const double H = std::get<0>(values);
        const Tensor<1, 2> dS = std::get<1>(values);
        const Tensor<1, 2> U = std::get<2>(values);
        P -= rho_ice * gravity * H * (dS * U) * dx;
      }

      for (unsigned int face = 0; face < faces_per_cell; ++face)
      {
        if (at_boundary_complement(std::get<0>(cell), face, dirichlet_ids))
        {
          assembly_face_data.reinit(cell, face);

          for (unsigned int q = 0; q < face_quad.size(); ++q)
          {
            const double dl = assembly_face_data.JxW(q);
            const auto values = assembly_face_data.values(q);
            const double H = std::get<0>(values);
            const double S = std::get<1>(values);
            const Tensor<1, 2> U = std::get<2>(values);
            const Tensor<1, 2> n = assembly_face_data.normal_vector(q);
            const double D = S - H;

            const double tau_I = 0.5 * gravity * rho_ice * H * H;
            const double tau_W = 0.5 * gravity * (D < 0) * rho_water * D * D;

            P += (tau_I - tau_W) * (U * n) * dl;
          }
        }
      }

    }

    return P;
  }


  DualVectorField<2> Gravity::derivative(
    const Field<2>& h,
    const Field<2>& s,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    const auto& discretization = get_discretization(h, s);
    DualVectorField<2> tau(discretization.shared_from_this());

    const auto quad = discretization.quad();
    auto assembly_data =
      make_assembly_data<2>(evaluate::function(h), evaluate::gradient(s));

    const auto face_quad = discretization.face_quad();
    auto assembly_face_data =
      make_assembly_face_data<2>(evaluate::function(h), evaluate::function(s));

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
        const Tensor<1, 2> dS = std::get<1>(values);
        const Tensor<1, 2> tau = -rho_ice * gravity * H * dS;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, 2> phi =
            assembly_data.fe_values_view<1>().value(i, q);
          cell_derivative(i) += tau * phi * dx;
        }
      }

      for (unsigned int face = 0; face < faces_per_cell; ++face)
      {
        if (at_boundary_complement(std::get<0>(cell), face, dirichlet_ids))
        {
          assembly_face_data.reinit(cell, face);

          for (unsigned int q = 0; q < face_quad.size(); ++q)
          {
            const double dl = assembly_face_data.JxW(q);
            const auto values = assembly_face_data.values(q);
            const double H = std::get<0>(values);
            const double S = std::get<1>(values);
            const Tensor<1, 2> n = assembly_face_data.normal_vector(q);
            const double D = S - H;

            const double tau_I = 0.5 * gravity * rho_ice * H * H;
            const double tau_W = 0.5 * gravity * (D < 0) * rho_water * D * D;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, 2> phi =
                assembly_face_data.fe_values_view<1>().value(i, q);
              cell_derivative(i) += (tau_I - tau_W) * (phi * n) * dl;
            }
          }
        }
      }

      std::get<1>(cell)->get_dof_indices(local_dof_ids);
      constraints.distribute_local_to_global(
        cell_derivative, local_dof_ids, tau.coefficients());
    }

    return tau;
  }


  double Gravity::derivative(
    const Field<2>& h,
    const Field<2>& s,
    const VectorField<2>& u
  ) const
  {
    return action(h, s, u);
  }


  IceStream::IceStream(
    const std::set<dealii::types::boundary_id>& dirichlet_ids_,
    const Viscosity& viscosity_,
    const Friction& friction_,
    const double tolerance_
  ) :
    dirichlet_ids(dirichlet_ids_),
    gravity(Gravity(dirichlet_ids)),
    viscosity(viscosity_),
    friction(friction_),
    tolerance(tolerance_)
  {}

  VectorField<2> IceStream::solve(
    const Field<2>& h,
    const Field<2>& s,
    const Field<2>& theta,
    const Field<2>& beta,
    const VectorField<2>& u0,
    const SolveOptions& options
  ) const
  {
    const auto& discretization = get_discretization(h, s, theta, u0);
    const dealii::ConstraintMatrix constraints =
      discretization.vector().make_constraints(dirichlet_ids);

    const auto F =
      [&](const VectorField<2>& v) -> double
      {
        return viscosity.action(h, theta, v) +
          friction.action(beta, v) -
          gravity.action(h, s, v);
      };

    const auto dF =
      [&](const VectorField<2>& v, const VectorField<2>& q) -> double
      {
        return viscosity.derivative(h, theta, v, q) +
          friction.derivative(beta, v, q) -
          gravity.derivative(h, s, q);
      };

    const auto P =
      [&](const VectorField<2>& v) -> VectorField<2>
      {
        VectorField<2> p(discretization.shared_from_this());

        DualVectorField<2> df =
          viscosity.derivative(h, theta, v, constraints) +
          friction.derivative(beta, v, constraints) -
          gravity.derivative(h, s, constraints);

        dealii::SparseMatrix<double> A =
          viscosity.hessian(h, theta, v, constraints);
        A.add(1.0, friction.hessian(beta, v, constraints));

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

}

