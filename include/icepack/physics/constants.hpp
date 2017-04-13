
#ifndef ICEPACK_CONSTANTS_HPP
#define ICEPACK_CONSTANTS_HPP

namespace icepack
{
  namespace constants
  {
    /// Some physical constants. These are all in units of megapascals, meters,
    /// years -- most interesting quantities in glaciology are around 1 in this
    /// system of units.
    constexpr double year = 365.25 * 24 * 3600;
    constexpr double gravity = 9.81 * year * year;              // m / yr^2
    constexpr double ideal_gas = 8.3144621e-3;                  // kJ / mol K
    constexpr double rho_ice = 917 / (year * year) * 1.0e-6;
    constexpr double rho_water = 1024 / (year * year) * 1.0e-6;
  }

  /// Compute the rate factor `A` in Glen's flow law for a given temperature.
  double rate_factor(const double temperature);
}

#endif
