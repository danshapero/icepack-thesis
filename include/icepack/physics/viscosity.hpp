
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
   * This auxiliary class is used to compute the membrane stress at a point
   * from the depth-averaged ice strain rate. The membrane stress \f$M\f$ is
   * \f[
   *   M = 2\mu(\dot\varepsilon + \mathrm{tr}(\dot\varepsilon)I),
   * \f]
   * where \f$\mu\f$ is the ice viscosity, \f$\dot\varepsilon\f$ is the strain
   * rate tensor, and \f$I\f$ is the identity tensor.
   *
   * The viscosity is in turn a function of the ice strain rate:
   * \f[
   *   \mu = \frac{A(T)^{-1/n}}{2}\sqrt{\frac{\dot\varepsilon : \dot\varepsilon + \mathrm{tr}(\dot\varepsilon)^2}{2}}^{1/n - 1}
   * \f]
   * where \f$A\f$ is the rate factor, \f$T\f$ is the temperature, and
   * \f$n = 3\f$ is the Glen flow law exponent.
   *
   * Usually the rate factor \f$A\f$ is defined as an Arrhenius function of the
   * ice temperature:
   * \f[
   *   A(T) = A_0\exp(-Q/RT).
   * \f]
   * However, depending on the application, it may be more convenient to use a
   * different parameterization and a field other than temperature. For
   * example, the \f$1/T\f$ dependence is numerically challenging when solving
   * inverse problems. Instead, one might prefer to use the inverse temperature
   * \f$\beta = QR/T\f$, in which case \f$A(\beta) = A_0e^\beta\f$. You can
   * pass your own parameterization of the rate factor to the constructor of
   * this class.
   *
   * @ingroup physics
   */
  struct MembraneStress
  {
    /**
     * Type alias for the parameterization of the rate factor
     */
    using RateFactor = double (*)(const double);

    /**
     * Create a membrane stress object using the default parameterization of
     * the rate factor as a function of temperature
     */
    MembraneStress(const double n = 3.0);

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
   * @brief Computes the net power and stress from viscosity as a function ice
   * thickness, temperature, and velocity
   *
   * The viscous power dissipation from internal ice shearing is
   * \f[
   *   P = \frac{n}{n + 1}\int_\Omega h M : \dot\varepsilon \hspace{1pt}dx
   * \f]
   * where \f$n\f$ is the Glen flow law exponent, \f$h\f$ is the ice thickness,
   * \f$M\f$ is the membrane stress (see the class `MembraneStress`), and
   * \f$\dot\varepsilon\f$ is the strain rate tensor.
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

