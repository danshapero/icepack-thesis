
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


  using DefaultFlags::flags;
  using DefaultFlags::face_flags;

  constexpr unsigned int faces_per_cell =
    dealii::GeometryInfo<2>::faces_per_cell;


  MassTransportImplicit::MassTransportImplicit()
  {}


  dealii::SparseMatrix<double> MassTransportImplicit::flux_matrix(
    const VectorField<2>& u,
    const dealii::ConstraintMatrix& constraints
  ) const
  {
    const auto& discretization = u.discretization();
    dealii::SparseMatrix<double> F(discretization.scalar().sparsity_pattern());

    const dealii::QGauss<2> quad = discretization.quad();
    const dealii::QGauss<1> face_quad = discretization.face_quad();

    auto assembly_data =
      make_assembly_data<2>(evaluate::function(u));
    auto assembly_face_data =
      make_assembly_face_data<2>(evaluate::function(u));

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

      for (unsigned int face = 0; face < faces_per_cell; ++face)
      {
        if (at_boundary(std::get<0>(cell), face))
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
      constraints.distribute_local_to_global(cell_matrix, local_dof_ids, F);
    }

    return F;
  }


  Field<2> MassTransportImplicit::solve(
    const double dt,
    const Field<2>& h0,
    const Field<2>& a,
    const VectorField<2>& u,
    const std::set<dealii::types::boundary_id>& ids,
    const Field<2>& h_inflow
  ) const
  {
    const auto& discretization = get_discretization(h0, a, u);
    const auto& scalar = discretization.scalar();
    const auto& M = scalar.mass_matrix();
    const auto& dh = scalar.dof_handler();

    const dealii::ZeroFunction<2> zero;
    std::map<dealii::types::boundary_id, const dealii::Function<2>*> fmap;
    for (const auto& id: ids)
      fmap[id] = &zero;
    std::map<dealii::types::global_dof_index, double> boundary_values;
    dealii::VectorTools::interpolate_boundary_values(dh, fmap, boundary_values);
    for (const auto& p: boundary_values)
      boundary_values[p.first] = h_inflow.coefficient(p.first);

    DualField<2> r = transpose(Field<2>(h0 + dt * a));
    dealii::Vector<double>& R = r.coefficients();

    Field<2> h(discretization);
    dealii::Vector<double>& H = h.coefficients();

    dealii::SparseMatrix<double> F = flux_matrix(u);
    F *= dt;
    F.add(1.0, M);

    dealii::MatrixTools::apply_boundary_values(boundary_values, F, H, R);
    scalar.constraints().condense(F, R);

    dealii::SparseDirectUMFPACK G;
    G.initialize(F);

    G.vmult(H, R);
    scalar.constraints().distribute(H);

    return h;
  }

}

