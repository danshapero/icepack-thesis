
#ifndef ICEPACK_SSA_HPP
#define ICEPACK_SSA_HPP

#include <deal.II/base/symmetric_tensor.h>
#include <icepack/field.hpp>

namespace icepack
{
  double rate_factor(const double temperature);


  struct ViscousRheology
  {
    ViscousRheology(const double n = 3.0);

    virtual double operator()(const double theta) const;
    virtual double dtheta(const double theta) const;

    const double n;
  };


  using dealii::SymmetricTensor;

  struct MembraneStress
  {
    MembraneStress(const ViscousRheology& rheology);

    virtual SymmetricTensor<2, 2> operator()(
      const double theta,
      const SymmetricTensor<2, 2> eps
    ) const;

    virtual SymmetricTensor<2, 2> dtheta(
      const double theta,
      const SymmetricTensor<2, 2> eps
    ) const;

    virtual SymmetricTensor<4, 2> du(
      const double theta,
      const SymmetricTensor<2, 2> eps
    ) const;

    const ViscousRheology& rheology;
  };


  struct Viscosity
  {
    const ViscousRheology rheology;
    const MembraneStress membrane_stress;

    Viscosity(const ViscousRheology& rheology = ViscousRheology());

    virtual double
    action(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity
    ) const;

    virtual DualVectorField<2>
    derivative(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity
    ) const;

    virtual dealii::SparseMatrix<double>
    hessian(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity
    ) const;
  };


  struct Gravity
  {
    Gravity();

    virtual double
    action(
      const Field<2>& thickness,
      const VectorField<2>& velocity
    ) const;

    virtual DualVectorField<2> derivative(const Field<2>& thickness) const;
  };

} // namespace icepack

#endif
