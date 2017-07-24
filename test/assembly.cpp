
#include <icepack/assembly.hpp>
#include <icepack/field.hpp>
#include "testing.hpp"

#include <typeinfo>
#include <iostream>

using icepack::testing::AffineFunction;
using icepack::testing::AffineTensorFunction;

int main()
{
  dealii::Triangulation<2> tria = icepack::testing::example_mesh();
  icepack::Discretization<2> discretization(tria, 1);

  const dealii::QGauss<2> quad = discretization.quad();
  const dealii::QGauss<1> face_quad = discretization.face_quad();

  const AffineFunction Phi(1.0, dealii::Point<2>(-4.0, 8.0));
  const icepack::Field<2> phi = icepack::interpolate(discretization, Phi);

  const AffineFunction U1(-4.0, dealii::Point<2>(3.5, 9.6));
  const AffineFunction U2(-8.6, dealii::Point<2>(-0.1, -2.3));
  const AffineTensorFunction U(U1, U2);
  const icepack::VectorField<2> u = icepack::interpolate(discretization, U);


  TEST_SUITE("does it even compile?")
  {
    const auto phi_evaluator = icepack::evaluate::function(phi);
    const auto grad_evaluator = icepack::evaluate::gradient(phi);

    const auto shape = icepack::shape_functions<2>::scalar::value();
    const auto shape_grad = icepack::shape_functions<2>::scalar::gradient();

    std::cout << typeid(shape.method).name() << "\n"
              << typeid(shape_grad.method).name() << "\n";
  }


  TEST_SUITE("more fields")
  {
    const auto assembly_data = icepack::make_assembly_data<2>(
      icepack::evaluate::function(phi),
      icepack::evaluate::function(u)
    );

    const auto assembly_face_data = icepack::make_assembly_face_data<2>(
      icepack::evaluate::function(phi),
      icepack::evaluate::function(u)
    );

    std::tuple<double, dealii::Tensor<1, 2>> t;
    CHECK(typeid(assembly_data.values(0)).name() == typeid(t).name());

    const auto assembly_data_copy = assembly_data;
    const auto assembly_face_data_copy = assembly_face_data;
  }


  TEST_SUITE("some derivatives")
  {
    const auto assembly_data = icepack::make_assembly_data<2>(
      icepack::evaluate::function(phi),
      icepack::evaluate::gradient(phi),
      icepack::evaluate::symmetric_gradient(u)
    );

    std::tuple<double, dealii::Tensor<1, 2>, dealii::SymmetricTensor<2, 2>> t;
    CHECK(typeid(assembly_data.values(0)).name() == typeid(t).name());
  }


  TEST_SUITE("iterating over a mesh")
  {
    auto assembly_data = icepack::make_assembly_data<2>(
      icepack::evaluate::function(phi),
      icepack::evaluate::symmetric_gradient(u)
    );

    const auto cell = discretization.begin();
    assembly_data.reinit(cell);

    double cell_max_phi_value = 0.0;
    double cell_max_eps_u_value = 0.0;
    for (size_t q = 0; q < quad.size(); ++q)
    {
      const std::tuple<double, dealii::SymmetricTensor<2, 2>> values =
        assembly_data.values(q);
      const double Phi = std::get<0>(values);
      const dealii::SymmetricTensor<2, 2> eps_u = std::get<1>(values);

      cell_max_phi_value = std::max(cell_max_phi_value, std::abs(Phi));
      cell_max_eps_u_value = std::max(cell_max_eps_u_value, eps_u.norm());

      CHECK(assembly_data.JxW(q) > 0);
    }

    CHECK(cell_max_phi_value > 0.0);
    CHECK(cell_max_eps_u_value > 0.0);

    auto assembly_face_data =
      icepack::make_assembly_face_data<2>(icepack::evaluate::function(phi));

    const size_t faces_per_cell = dealii::GeometryInfo<2>::faces_per_cell;
    for (unsigned int face = 0; face < faces_per_cell; ++face)
    {
      assembly_face_data.reinit(cell, face);

      for (size_t q = 0; q < face_quad.size(); ++q)
        CHECK(assembly_face_data.JxW(q) > 0);
    }
  }


  TEST_SUITE("shape functions")
  {
    auto assembly_data = icepack::make_assembly_data<2>(
      icepack::evaluate::function(phi),
      icepack::evaluate::gradient(phi),
      icepack::evaluate::symmetric_gradient(u)
    );

    const dealii::QGauss<2> quad = discretization.quad();
    const size_t n_q_points = quad.size();

    const auto cell = discretization.begin();
    assembly_data.reinit(cell);
    const auto& scalar_view = assembly_data.fe_values_view<0>();
    const auto& vector_view = assembly_data.fe_values_view<1>();

    TEST_SUITE("scalar")
    {
      const auto shape_grad = icepack::shape_functions<2>::scalar::gradient();

      const size_t n_dofs = discretization(0).finite_element().dofs_per_cell;
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int i = 0; i < n_dofs; ++i)
          CHECK(shape_grad(scalar_view, i, q).norm() > 0);
      }
    }


    TEST_SUITE("vector")
    {
      const auto grad_u = icepack::shape_functions<2>::vector::gradient();
      const auto eps_u =
        icepack::shape_functions<2>::vector::symmetric_gradient();

      const size_t n_dofs = discretization(1).finite_element().dofs_per_cell;
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int i = 0; i < n_dofs; ++i)
        {
          CHECK(grad_u(vector_view, i, q).norm() > 0);
          CHECK(eps_u(vector_view, i, q).norm() > 0);
        }
      }
    }
  }


  return 0;
}

