
#ifndef ICEPACK_VISCOSITY_HPP
#define ICEPACK_VISCOSITY_HPP

#include <icepack/field.hpp>
#include <deal.II/base/symmetric_tensor.h>

namespace icepack
{
  /**
   * Compute the rate factor `A` in Glen's flow law as a function of the ice
   * temperature. See Greve and Blatter or Cuffey and Paterson.
   */
  double rate_factor(const double temperature);


  /**
   * @brief Computes the stress in the 2D plane as a function of strain rate
   *
   * @ingroup physics
   */
  struct MembraneStress
  {
    /**
     * Create a membrane stress object using the default parameterization of
     * the rate factor as a function of temperature
     */
    MembraneStress(const double n = 3.0);

    typedef double (*RateFactor)(const double);

    /**
     * Create a membrane stress object using a user-supplied parameterization
     * of the rate factor
     */
    MembraneStress(const double n, const RateFactor A, const RateFactor dA);

    /**
     * Compute the rheological parameter as a function of the temperature
     */
    double B(const double theta) const;

    /**
     * Compute the derivative of the rheological parameter with respect to
     * temperature
     */
    double dB(const double theta) const;

    /**
     * Compute the membrane stress as a function of the local ice temperature
     * and strain rate
     */
    dealii::SymmetricTensor<2, 2> operator()(
      const double theta,
      const dealii::SymmetricTensor<2, 2> eps
    ) const;

    /**
     * Compute the derivative of the membrane stress with respect to the ice
     * temperature
     *
     * This method is used in solving inverse problems, where we need to
     * compute the derivative of the model physics with respect to temperature.
     */
    dealii::SymmetricTensor<2, 2> dtheta(
      const double theta,
      const dealii::SymmetricTensor<2, 2> eps
    ) const;

    /**
     * Compute the derivative of the membrane stress with respect to the ice
     * strain rate
     *
     * This method is used in computing the linearization of the forward model
     * physics, which we need to apply Newton's method.
     */
    dealii::SymmetricTensor<4, 2> du(
      const double theta,
      const dealii::SymmetricTensor<2, 2> eps
    ) const;

    /**
     * Glen flow law exponent
     */
    const double n;

  protected:
    /**
     * The parameterization of the rate factor as a function of the ice
     * temperature
     */
    RateFactor A;

    /**
     * The derivative of the parameterization of the rate factor
     */
    RateFactor dA;
  };


  /**
   * @brief Computes the net power and membrane stress as a function ice
   * thickness, temperature, and velocity
   *
   * @ingroup physics
   */
  struct Viscosity
  {
    /**
     * Constructor, takes in an object for computing the membrane stress. You
     * may want to change the parameterization for the membrane stress as a
     * function of some variable other than temperature, e.g. a combination of
     * temperature and damage, inverse temperature, etc., depending on the
     * application.
     */
    Viscosity(const MembraneStress& membrane_stress);

    /**
     * Evaluate the viscous power dissipation. The action for the shallow shelf
     * approximation is the sum of the viscous and gravitational power.
     */
    double action(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity
    ) const;

    /**
     * Evaluate the derivative of the viscous power dissipation; this is the
     * membrane stress field.
     */
    DualVectorField<2> derivative(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    /**
     * Evaluate the directional derivative of the viscous power dissipation
     * along some velocity field.
     */
    double derivative(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const VectorField<2>& direction
    ) const;

    /**
     * Evaluate the second derivative of the viscous power dissipation with
     * respect to the velocity.
     */
    dealii::SparseMatrix<double> hessian(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    const MembraneStress membrane_stress;
  };

}

#endif

