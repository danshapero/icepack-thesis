
#include <icepack/physics/mass_transport.hpp>
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
    const unsigned int n_q_points = quad.size();
    const unsigned int n_face_q_points = face_quad.size();

    std::vector<Tensor<1, 2>> u_values(n_q_points);
    std::vector<Tensor<1, 2>> u_face_values(n_face_q_points);

    const auto& h_fe = discretization.scalar().finite_element();
    const auto& u_fe = discretization.vector().finite_element();
    const unsigned int dofs_per_cell = h_fe.dofs_per_cell;
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    dealii::FEValues<2> h_fe_values(h_fe, quad, flags);
    dealii::FEValues<2> u_fe_values(u_fe, quad, flags);
    dealii::FEFaceValues<2> h_fe_face_values(h_fe, face_quad, face_flags);
    dealii::FEFaceValues<2> u_fe_face_values(u_fe, face_quad, face_flags);

    const auto& exs = h_fe_values[dealii::FEValuesExtractors::Scalar(0)];
    const auto& exv = u_fe_values[dealii::FEValuesExtractors::Vector(0)];
    const auto& exfs = h_fe_face_values[dealii::FEValuesExtractors::Scalar(0)];
    const auto& exfv = u_fe_face_values[dealii::FEValuesExtractors::Vector(0)];

    for (const auto& it: discretization)
    {
      const auto& its = std::get<0>(it);
      const auto& itv = std::get<1>(it);

      cell_matrix = 0;
      h_fe_values.reinit(its);
      u_fe_values.reinit(itv);

      exv.get_function_values(u.coefficients(), u_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double dx = h_fe_values.JxW(q);
        const Tensor<1, 2> U = u_values[q];

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, 2> grad_phi_i = exs.gradient(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            const double phi_j = exs.value(j, q);
            cell_matrix(i, j) -= grad_phi_i * U * phi_j * dx;
          }
        }
      }

      for (unsigned int face = 0; face < faces_per_cell; ++face)
      {
        if (at_boundary(its, face))
        {
          h_fe_face_values.reinit(its, face);
          u_fe_face_values.reinit(itv, face);

          exfv.get_function_values(u.coefficients(), u_face_values);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            const double dl = h_fe_face_values.JxW(q);
            const Tensor<1, 2> U = u_face_values[q];
            const Tensor<1, 2> n = h_fe_face_values.normal_vector(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double phi_i = exfs.value(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const double phi_j = exfs.value(j, q);
                cell_matrix(i, j) += phi_i * phi_j * U * n * dl;
              }
            }
          }
        }
      }

      std::get<0>(it)->get_dof_indices(local_dof_ids);
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

