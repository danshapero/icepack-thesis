
#include <icepack/field.hpp>
#include <deal.II/numerics/vector_tools.h>

namespace icepack
{
  using dealii::Function;
  using dealii::TensorFunction;

  /* -----------------------------------------
   * FieldType member function implementations
   * ----------------------------------------- */

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>::FieldType(const Discretization<dim>& dsc)
    : discretization_(&dsc),
      coefficients_(discretization()(rank).dof_handler().n_dofs())
  {
    coefficients_ = 0;
  }

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>::FieldType(FieldType<rank, dim, duality>&& phi)
    : discretization_(phi.discretization_),
      coefficients_(std::move(phi.coefficients_))
  {
    phi.discretization_ = nullptr;
    phi.coefficients_.reinit(0);
  }

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  FieldType<rank, dim, duality>::operator=(FieldType<rank, dim, duality>&& phi)
  {
    discretization_ = phi.discretization_;
    coefficients_ = std::move(phi.coefficients_);

    phi.discretization_ = nullptr;
    phi.coefficients_.reinit(0);

    return *this;
  }

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  FieldType<rank, dim, duality>::
  operator=(const FieldType<rank, dim, duality>& phi)
  {
    discretization_ = phi.discretization_;
    coefficients_ = phi.coefficients_;

    return *this;
  }

  template <int rank, int dim, Duality duality>
  const Discretization<dim>&
  FieldType<rank, dim, duality>::discretization() const
  {
    return *discretization_;
  }

  template <int rank, int dim, Duality duality>
  const Vector<double>&
  FieldType<rank, dim, duality>::coefficients() const
  {
    return coefficients_;
  }

  template <int rank, int dim, Duality duality>
  Vector<double>&
  FieldType<rank, dim, duality>::coefficients()
  {
    return coefficients_;
  }

  template <int rank, int dim, Duality duality>
  double
  FieldType<rank, dim, duality>::coefficient(const size_t i) const
  {
    return coefficients_[i];
  }


  // Explicitly instantiate a bunch of templates.
  template class FieldType<0, 2, primal>;
  template class FieldType<0, 3, primal>;
  template class FieldType<1, 2, primal>;
  template class FieldType<1, 3, primal>;
  template class FieldType<0, 2, dual>;
  template class FieldType<0, 3, dual>;
  template class FieldType<1, 2, dual>;
  template class FieldType<1, 3, dual>;



  /* --------------------------------------------------------
   * Interpolating functions to finite element representation
   * -------------------------------------------------------- */

  template <int dim>
  Field<dim> interpolate(
    const Discretization<dim>& discretization,
    const dealii::Function<dim>& phi
  )
  {
    Field<dim> Phi(discretization);
    const auto& dof_handler = discretization.scalar().dof_handler();
    dealii::VectorTools::interpolate(dof_handler, phi, Phi.coefficients());
    return Phi;
  }


  template <int dim>
  VectorField<dim>
  interpolate(
    const Discretization<dim>& discretization,
    const dealii::TensorFunction<1, dim>& phi
  )
  {
    VectorField<dim> Phi(discretization);
    const dealii::VectorFunctionFromTensorFunction<dim> vphi(phi);
    const auto& dof_handler = discretization.vector().dof_handler();
    dealii::VectorTools::interpolate(dof_handler, vphi, Phi.coefficients());
    return Phi;
  }


  template Field<2> interpolate(const Discretization<2>&, const Function<2>&);
  template Field<3> interpolate(const Discretization<3>&, const Function<3>&);

  template
  VectorField<2>
  interpolate(const Discretization<2>&, const TensorFunction<1, 2>&);

  template
  VectorField<3>
  interpolate(const Discretization<3>&, const TensorFunction<1, 3>&);


} // namespace icepack
