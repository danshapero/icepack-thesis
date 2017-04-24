
#ifndef ICEPACK_SSA_HPP
#define ICEPACK_SSA_HPP

#include <deal.II/base/symmetric_tensor.h>
#include <icepack/field.hpp>
#include <icepack/numerics/convergence_log.hpp>

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

    Viscosity(const ViscousRheology& rheology);

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


  struct IceShelf
  {
    IceShelf(
      const Viscosity& viscosity,
      const double convergence_tolerance = 1.0e-6
    );

    VectorField<2> solve(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const std::map<dealii::types::global_dof_index, double>& bcs,
      numerics::ConvergenceLog& convergence_log
    ) const;

    const Gravity gravity;
    const Viscosity viscosity;
    const double tolerance;
  };

} // namespace icepack

#endif
