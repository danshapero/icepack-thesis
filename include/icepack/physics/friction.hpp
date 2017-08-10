
#ifndef ICEPACK_FRICTION_HPP
#define ICEPACK_FRICTION_HPP

#include <icepack/field.hpp>
#include <deal.II/base/symmetric_tensor.h>

namespace icepack
{
  /**
   * @brief Computes the basal shear stress from ice sliding over its bed
   *
   * This auxiliary class is used to compute the shear stress at a poin from
   * ice sliding over its underlying bed. The basal shear stress is
   * \f[
   *   \tau = -C|u|^{1/m - 1}u,
   * \f]
   * where \f$C\f$ is the friction coefficient, \f$u\f$ is the sliding
   * velocity, and \f$m\f$ is the sliding rheology exponent. The default value
   * is \f$m = 3\f$.
   *
   * Usually the friction coefficient is defined as a function of some
   * parameter \f$\beta\f$. For example, \f$C = \beta^2\f$ is common in the
   * inverse problems literature to guarantee the positivity of the friction
   * coefficient. You can pass your own parameterization to the constructor of
   * this class; the default is just \f$C(\beta) = \beta\f$.
   *
   * @ingroup physics
   */
  struct BasalStress
  {
    /**
     * Type alias for the parameterization fo the friction coefficient
     */
    using FrictionCoefficient = double (*)(const double);

    /**
     * Create a basal stress object for a given rheological exponent
     */
    BasalStress(const double m = 3.0);

    /**
     * Create a basal stress object with a user-supplied parameterization for
     * the friction coefficient
     */
    BasalStress(
      const double m,
      const FrictionCoefficient C,
      const FrictionCoefficient dC
    );

    /**
     * Compute the basal shear stress from the basal friction parameter and the
     * sliding velocity
     */
    dealii::Tensor<1, 2> operator()(
      const double beta,
      const dealii::Tensor<1, 2> velocity
    ) const;

    /**
     * Compute the derivative of the basal shear stress with respect to the
     * basal friction parameter
     */
    dealii::Tensor<1, 2> dbeta(
      const double beta,
      const dealii::Tensor<1, 2> velocity
    ) const;

    /**
     * Compute the derivative of the basal shear stress with respect to the ice
     * velocity
     */
    dealii::SymmetricTensor<2, 2> du(
      const double beta,
      const dealii::Tensor<1, 2> velocity
    ) const;

    /**
     * Rheological exponent for ice sliding
     */
    const double m;

    /**
     * Function to compute the friction coefficient \f$C\f$ from the friction
     * parameter \f$\beta\f$
     */
    const FrictionCoefficient C;

    /**
     * Function to compute the derivative of the friction coefficient with
     * respect to \f$\beta\f$
     */
    const FrictionCoefficient dC;
  };


  /**
   * @brief Compute the net power and stress from basal friction as a function
   * of the friction parameter and sliding velocity
   *
   * The power dissipation due to sliding friction is
   * \f[
   *   P = \frac{m}{m + 1}\int_\Omega\tau_b\cdot u\hspace{1pt}dx
   * \f]
   * where \f$m\f$ is the sliding rheology exponent, \f$\tau_b\f$ is the basal
   * shear stress (see the class `BasalStress`), and \f$u\f$ is the ice sliding
   * velocity.
   *
   * @ingroup physics
   */
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

