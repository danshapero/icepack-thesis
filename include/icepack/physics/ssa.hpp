
#ifndef ICEPACK_SSA_HPP
#define ICEPACK_SSA_HPP

#include <deal.II/base/symmetric_tensor.h>
#include <icepack/field.hpp>
#include <icepack/numerics/optimization.hpp>

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
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    virtual double
    derivative(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const VectorField<2>& direction
    ) const;

    virtual dealii::SparseMatrix<double>
    hessian(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
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

    virtual DualVectorField<2>
    derivative(
      const Field<2>& thickness,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    virtual double
    derivative(
      const Field<2>& thickness,
      const VectorField<2>& direction
    ) const;
  };


  struct IceShelf
  {
    IceShelf(
      const std::set<dealii::types::boundary_id>& dirichlet_boundary_ids = {0},
      const Viscosity& viscosity = Viscosity(ViscousRheology()),
      const double convergence_tolerance = 1.0e-6
    );

    using SolveOptions = numerics::NewtonSearchOptions<VectorField<2>>;

    VectorField<2> solve(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const SolveOptions solve_options = SolveOptions()
    ) const;

    VectorField<2> solve(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const std::set<dealii::types::boundary_id>& dirichlet_boundary_ids,
      const SolveOptions solve_options = SolveOptions()
    ) const;

    const Gravity gravity;
    const Viscosity viscosity;
    const std::set<dealii::types::boundary_id> dirichlet_boundary_ids;
    const double tolerance;
  };

} // namespace icepack

#endif
