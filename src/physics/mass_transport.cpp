
#include <icepack/physics/mass_transport.hpp>
#include <icepack/assembly.hpp>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

namespace icepack
{
  using dealii::Tensor;

  double compute_timestep(const double C, const VectorField<2>& u)
  {
    const auto& tria = u.discretization().triangulation();
    const double dx = dealii::GridTools::minimal_cell_diameter(tria);
    const double U = max(u);
    return C * dx / U;
  }


  constexpr unsigned int faces_per_cell =
    dealii::GeometryInfo<2>::faces_per_cell;


  MassTransport::MassTransport(
    const std::set<dealii::types::boundary_id>& inflow_boundary_ids_
  ) :
    inflow_boundary_ids(inflow_boundary_ids_)
  {}


  dealii::SparseMatrix<double> MassTransport::flux_matrix(
    const VectorField<2>& u
  ) const
  {
    const auto& discretization = u.discretization();
    dealii::SparseMatrix<double> F(discretization.scalar().sparsity_pattern());

    const dealii::QGauss<2> quad = discretization.quad();
    auto assembly_data =
      make_assembly_data<2>(evaluate::function(u));

    const auto& h_fe = discretization.scalar().finite_element();
    const unsigned int dofs_per_cell = h_fe.dofs_per_cell;
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    for (const auto& cell: discretization)
    {
      cell_matrix = 0;
      assembly_data.reinit(cell);

      for (unsigned int q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const Tensor<1, 2> U = std::get<0>(assembly_data.values(q));

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, 2> grad_phi_i =
            assembly_data.fe_values_view<0>().gradient(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            const double phi_j = assembly_data.fe_values_view<0>().value(j, q);
            cell_matrix(i, j) -= grad_phi_i * U * phi_j * dx;
          }
        }
      }

      std::get<0>(cell)->get_dof_indices(local_dof_ids);
      F.add(local_dof_ids, cell_matrix);
    }

    return F;
  }


  dealii::SparseMatrix<double> MassTransport::boundary_flux_matrix(
    const VectorField<2>& u,
    const std::set<dealii::types::boundary_id>& boundary_ids
  ) const
  {
    const auto& discretization = u.discretization();
    dealii::SparseMatrix<double> F(discretization.scalar().sparsity_pattern());

    const dealii::QGauss<1> face_quad = discretization.face_quad();
    auto assembly_face_data =
      make_assembly_face_data<2>(evaluate::function(u));

    const auto& h_fe = discretization.scalar().finite_element();
    const unsigned int dofs_per_cell = h_fe.dofs_per_cell;
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    for (const auto& cell: discretization)
    {
      cell_matrix = 0;

      for (unsigned int face = 0; face < faces_per_cell; ++face)
      {
        if (at_boundary_complement(std::get<0>(cell), face, boundary_ids))
        {
          assembly_face_data.reinit(cell, face);

          for (unsigned int q = 0; q < face_quad.size(); ++q)
          {
            const double dl = assembly_face_data.JxW(q);
            const Tensor<1, 2> U = std::get<0>(assembly_face_data.values(q));
            const Tensor<1, 2> n = assembly_face_data.normal_vector(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double phi_i =
                assembly_face_data.fe_values_view<0>().value(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const double phi_j =
                  assembly_face_data.fe_values_view<0>().value(j, q);
                cell_matrix(i, j) += phi_i * phi_j * U * n * dl;
              }
            }
          }
        }
      }

      std::get<0>(cell)->get_dof_indices(local_dof_ids);
      F.add(local_dof_ids, cell_matrix);
    }

    return F;
  }


  DualField<2> MassTransport::boundary_flux(
    const Field<2>& h,
    const VectorField<2>& u,
    const std::set<dealii::types::boundary_id>& boundary_ids
  ) const
  {
    const auto& discretization = get_discretization(h, u);
    DualField<2> f(discretization.shared_from_this());

    const dealii::QGauss<1> face_quad = discretization.face_quad();
    auto assembly_face_data =
      make_assembly_face_data<2>(evaluate::function(h), evaluate::function(u));

    const auto& h_fe = discretization.scalar().finite_element();
    const unsigned int dofs_per_cell = h_fe.dofs_per_cell;
    dealii::Vector<double> cell_values(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    for (const auto& cell: discretization)
    {
      cell_values = 0;

      for (unsigned int face = 0; face < faces_per_cell; ++face)
      {
        if (at_boundary(std::get<0>(cell), face, boundary_ids))
        {
          assembly_face_data.reinit(cell, face);

          for (unsigned int q = 0; q < face_quad.size(); ++q)
          {
            const double dl = assembly_face_data.JxW(q);
            const auto values = assembly_face_data.values(q);
            const double H = std::get<0>(values);
            const Tensor<1, 2> U = std::get<1>(values);
            const Tensor<1, 2> n = assembly_face_data.normal_vector(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double phi =
                assembly_face_data.fe_values_view<0>().value(i, q);
              cell_values(i) += phi * H * U * n * dl;
            }
          }
        }
      }

      std::get<0>(cell)->get_dof_indices(local_dof_ids);
      f.coefficients().add(local_dof_ids, cell_values);
    }

    return f;
  }


  Field<2> MassTransport::solve(
    const double dt,
    const Field<2>& h0,
    const Field<2>& a,
    const VectorField<2>& u,
    const Field<2>& h_inflow
  ) const
  {
    const auto& discretization = get_discretization(h0, a, u);
    const auto& scalar = discretization.scalar();
    const auto& M = scalar.mass_matrix();

    DualField<2> f_i = boundary_flux(h_inflow, u, inflow_boundary_ids);
    DualField<2> r = transpose(Field<2>(h0 + dt * a)) - dt * f_i;
    dealii::Vector<double>& R = r.coefficients();

    Field<2> h(discretization.shared_from_this());
    dealii::Vector<double>& H = h.coefficients();

    dealii::SparseMatrix<double> F = flux_matrix(u);
    const dealii::SparseMatrix<double> F_o =
      boundary_flux_matrix(u, inflow_boundary_ids);
    F.add(1.0, F_o);
    F *= dt;
    F.add(1.0, M);

    scalar.constraints().condense(F, R);
    dealii::SparseDirectUMFPACK G;
    G.initialize(F);

    G.vmult(H, R);
    scalar.constraints().distribute(H);

    return h;
  }

}

