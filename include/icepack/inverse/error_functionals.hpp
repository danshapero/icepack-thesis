
#ifndef ICEPACK_ERROR_FUNCTIONALS_HPP
#define ICEPACK_ERROR_FUNCTIONALS_HPP

#include <icepack/field.hpp>

namespace icepack
{
  template <int dim>
  struct MeanSquareError
  {
    double action(
      const VectorField<dim>& u,
      const VectorField<dim>& v,
      const Field<dim>& sigma
    ) const;

    DualVectorField<dim> derivative(
      const VectorField<dim>& u,
      const VectorField<dim>& v,
      const Field<dim>& sigma,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;
  };
}

#endif
