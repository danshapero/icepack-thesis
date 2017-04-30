
#ifndef ICEPACK_GRID_DATA_HPP
#define ICEPACK_GRID_DATA_HPP

#include <deal.II/base/function_lib.h>

namespace icepack
{
  /**
   * Class for reading gridded data into a `dealii::Function` object
   */
  class GridData :
    public dealii::Functions::InterpolatedTensorProductGridData<2>
  {
  public:
    /**
     * Construct a GridData object from arrays giving the locations of the
     * measurements and the data.
     * You will probably never need to use this method; instead, use the
     * functions defined below to read various common data formats into a
     * GridData object.
     */
    GridData(
      const std::array<std::vector<double>, 2>& coodinate_values,
      const dealii::Table<2, double>& data_values,
      const double missing_data_value
    );

    /**
     * Return the value of the gridded data near a point, or, if no suitable
     * value can be interpolated to that point due to missing data, return the
     * missing data value.
     */
    virtual double
    value(const dealii::Point<2>& x, const unsigned int = 0) const;

    /**
     * Return whether or not a given point is masked, i.e. one of the data
     * points necessary to interpolate a value to this point is missing from
     * the observations.
     */
    bool is_masked(const dealii::Point<2>& x) const;

    /**
     * Horizontal extent of the gridded data
     */
    const std::array<double, 2> xrange;

    /**
     * Vertical extent of the gridded data
     */
    const std::array<double, 2> yrange;

    /**
     * Value to indicate missing data
     */
    const double missing_data_value;

    /**
     * Table to describe where there are missing measurements
     */
    const dealii::Table<2, bool> mask;
  };


  /**
   * Read a gridded data set stored in the ESRI Arc/Info ASCII grid format. See
   *   http://en.wikipedia.org/wiki/Esri_grid
   * for format specification and more info.
   */
  GridData readArcAsciiGrid(const std::string& filename);

}

#endif
