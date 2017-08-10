
#ifndef ICEPACK_FRICTION_HPP
#define ICEPACK_FRICTION_HPP

#include <icepack/field.hpp>
#include <deal.II/base/symmetric_tensor.h>

namespace icepack
{
  struct BasalStress
  {
    BasalStress(const double m = 3.0);

    using FrictionCoefficient = double (*)(const double);

    BasalStress(
      const double m,
      const FrictionCoefficient C,
      const FrictionCoefficient dC
    );

    const double m;
    const FrictionCoefficient C;
    const FrictionCoefficient dC;

    dealii::Tensor<1, 2> operator()(
      const double beta,
      const dealii::Tensor<1, 2> velocity
    ) const;

    dealii::Tensor<1, 2> dbeta(
      const double beta,
      const dealii::Tensor<1, 2> velocity
    ) const;

    dealii::SymmetricTensor<2, 2> du(
      const double beta,
      const dealii::Tensor<1, 2> velocity
    ) const;
  };


  struct Friction
  {
    Friction(const BasalStress& basal_stress);

    double action(
      const Field<2>& beta,
      const VectorField<2>& velocity
    ) const;

    DualVectorField<2> derivative(
      const Field<2>& beta,
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    double derivative(
      const Field<2>& beta,
      const VectorField<2>& velocity,
      const VectorField<2>& direction
    ) const;

    dealii::SparseMatrix<double> hessian(
      const Field<2>& beta,
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    const BasalStress basal_stress;
  };
}

#endif

