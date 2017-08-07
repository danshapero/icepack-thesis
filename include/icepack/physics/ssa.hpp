
#ifndef ICEPACK_SSA_HPP
#define ICEPACK_SSA_HPP

#include <icepack/field.hpp>
#include <icepack/numerics/optimization.hpp>
#include <deal.II/base/symmetric_tensor.h>

namespace icepack
{
  /**
   * Compute the rate factor `A` in Glen's flow law as a function of the ice
   * temperature. See Greve and Blatter or Cuffey and Paterson.
   */
  double rate_factor(const double temperature);


  /**
   * @brief Computes the rheological parameter of the ice as a function of
   * temperature
   *
   * In order to solve for the ice velocity, we need to be able to evaluate the
   * rheology, but to solve inverse problems, we also need to be able to
   * evaluate the derivative of the rheology as a function of temperature too.
   * Consequently, these two functions are packed into an object.
   *
   * @ingroup physics
   */
  struct ViscousRheology
  {
    /**
     * Create a rheology object for a given value of the Glen flow law
     * exponent, which defaults to 3.0.
     */
    ViscousRheology(const double n = 3.0);

    /**
     * Evaluate the rheology coefficient for a given temperature.
     */
    virtual double operator()(const double theta) const;

    /**
     * Evaluate the derivative of the rheology with respect to temperature.
     * (This is used in solving inverse problems.)
     */
    virtual double dtheta(const double theta) const;

    const double n;
  };


  using dealii::SymmetricTensor;


  /**
   * @brief Computes the stress in the 2D plane as a function of strain rate
   *
   * @ingroup physics
   */
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

    const ViscousRheology rheology;
  };


  /**
   * @brief Computes the net power and membrane stress as a function ice
   * thickness, temperature, and velocity
   *
   * @ingroup physics
   */
  struct Viscosity
  {
    const MembraneStress membrane_stress;

    /**
     * Constructor, takes in an object for computing the rheology. You may want
     * to change the rheology object to use a different parameterization for
     * the rheology as a function of some variable other than temperature, e.g.
     * a combination of temperature and damage, inverse temperature, etc.
     */
    Viscosity(const ViscousRheology& rheology);

    /**
     * Evaluate the viscous power dissipation. The action for the shallow shelf
     * approximation is the sum of the viscous and gravitational power.
     */
    virtual double
    action(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity
    ) const;

    /**
     * Evaluate the derivative of the viscous power dissipation; this is the
     * membrane stress field.
     */
    virtual DualVectorField<2>
    derivative(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    /**
     * Evaluate the directional derivative of the viscous power dissipation
     * along some velocity field.
     */
    virtual double
    derivative(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const VectorField<2>& direction
    ) const;

    /**
     * Evaluate the second derivative of the viscous power dissipation with
     * respect to the velocity.
     */
    virtual dealii::SparseMatrix<double>
    hessian(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;
  };


  /**
   * @brief Computes the gravitational power dissipation and stress for shallow
   * shelf flows
   *
   * @ingroup physics
   */
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


  /**
   * @brief Class for solving for the ice velocity and thickness, under the
   * shallow shelf approximation
   *
   * @ingroup physics
   */
  struct IceShelf
  {
    /**
     * Construct an ice shelf model object, given the IDs of the parts of the
     * ice shelf boundary that correspond to Dirichlet boundary conditions.
     */
    IceShelf(
      const std::set<dealii::types::boundary_id>& dirichlet_boundary_ids,
      const Viscosity& viscosity = Viscosity(ViscousRheology()),
      const double convergence_tolerance = 1.0e-6
    );

    using SolveOptions = numerics::NewtonSearchOptions<VectorField<2>>;

    /**
     * Compute the ice velocity for a given thickness and temperature starting
     * from some initial guess, using the Dirichlet boundaries that were set
     * when the present `IceShelf` object was initialized. The boundary
     * conditions for the velocity are obtained from the initial guess.
     */
    VectorField<2> solve(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const SolveOptions solve_options = SolveOptions()
    ) const;

    const std::set<dealii::types::boundary_id> dirichlet_boundary_ids;

    const Gravity gravity;
    const Viscosity viscosity;
    const double tolerance;
  };

} // namespace icepack

#endif
