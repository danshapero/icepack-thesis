
#ifndef ICEPACK_GRID_DATA_HPP
#define ICEPACK_GRID_DATA_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/table.h>
#include <boost/optional.hpp>

namespace icepack
{
  /**
   * @brief Class for reading gridded data into a `dealii::Function` object.
   */
  template <int dim>
  class GridData : public dealii::Function<dim>
  {
  public:
    /**
     * Construct a gridded data set from the lower left and upper right points,
     * and tables for the data and mask
     */
    GridData(
      const dealii::Point<dim>& lower,
      const dealii::Point<dim>& upper,
      const dealii::Table<dim, double>& data,
      const dealii::Table<dim, bool>& mask
    );

    /**
     * Return a double if there is measured data at the given point, or nothing
     * if the data around the point is missing
     */
    boost::optional<double> operator()(const dealii::Point<dim>& x) const;

    /**
     * Return the value of the gridded data at a point, or throw an exception
     * if there is no data
     */
    double
    value(const dealii::Point<dim>& x, const unsigned int = 0) const override;

    /**
     * Table with the data values, or `NaN` where there is no data
     */
    const dealii::Table<dim, double> data;

    /**
     * Return whether or not a given point is masked, i.e. one of the data
     * points necessary to interpolate a value to this point is missing from
     * the observations.
     */
    bool is_masked(const dealii::Point<dim>& x) const;

    /**
     * Table to describe where there are missing measurements
     */
    const dealii::Table<dim, bool> mask;

    /**
     * Return the number of intervals along each coordinate direction
     */
    std::array<size_t, dim> n_subintervals() const;

    /**
     * Return the coordinates of a given grid point
     */
    dealii::Point<dim> grid_point(const std::array<size_t, dim>& index) const;

    /**
     * Return the size of a single cell of the grid
     */
    dealii::Point<dim> cell_size() const;

  protected:
    const dealii::Point<dim> lower;
    const dealii::Point<dim> upper;

    dealii::TableIndices<dim> table_index(const dealii::Point<dim>& x) const;

    dealii::Point<dim> local_coords(
      const dealii::Point<dim>& x,
      const dealii::TableIndices<dim>& index
    ) const;
  };


  /**
   * Read a gridded data set stored in the ESRI Arc/Info ASCII grid format. See
   * the [Wikipedia article](http://en.wikipedia.org/wiki/Esri_grid) for the
   * format specification and more info.
   */
  GridData<2> read_arc_ascii_grid(const std::string& filename);


  /**
   * Write out a gridded data set using the ESRI Arc/Info ASCII grid format.
   */
  void write_arc_ascii_grid(
    const GridData<2>& grid,
    const std::string& filename,
    const double missing_data_value
  );
}

#endif
