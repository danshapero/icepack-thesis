
#include <fstream>
#include <icepack/grid_data.hpp>
#include "testing.hpp"

int main()
{
  TEST_SUITE("manually creating gridded data sets")
  {
    // Make the coordinate values
    const size_t nx = 128, ny = 192;
    const double dx = 2.0, dy = 2.0;
    const double x0 = 3.0, y0 = 7.0;

    std::vector<double> xs(nx), ys(ny);

    for (size_t i = 0; i < ny; ++i)
      ys[i] = y0 + i * dy;

    for (size_t j = 0; j < nx; ++j)
      xs[j] = x0 + j * dx;

    std::array<std::vector<double>, 2> coordinate_values = {{xs, ys}};

    // Make some synthetic data
    dealii::Table<2, double> data(nx, ny);

    for (size_t i = 0; i < ny; ++i)
      for (size_t j = 0; j < nx; ++j)
        data[j][i] = xs[j] + ys[i];

    // Make one of the measurements a missing data point
    const double missing = -9999.0;
    const size_t I = ny / 2, J = nx / 2;
    data[J][I] = missing;

    // Make the gridded data object
    const icepack::GridData grid_data(coordinate_values, data, missing);

    // Check that a point near the missing data is masked
    const dealii::Point<2> x(x0 + (J + 0.5) * dx, y0 + (I + 0.5) * dy);
    CHECK(grid_data.is_masked(x));
    CHECK(grid_data.value(x) <= missing);

    const dealii::Point<2> y(x0 + dx, y0 + dy);
    CHECK(not grid_data.is_masked(y));
    CHECK_REAL(grid_data.value(y), y[0] + y[1], 1.0e-16);
  }


  TEST_SUITE("reading Arc ASCII grids")
  {
    const size_t nx = 11, ny = 6;
    const double xo = 0.0, yo = 0.0;
    const double dx = 0.2, dy = dx;
    const double missing = -9999.0;

    const std::string filename = "example_arc_grid.txt";

    // Create an Arc ASCII grid file.
    {

      std::ofstream fid(filename);
      fid << "ncols          " << nx << "\n"
          << "nrows          " << ny << "\n"
          << "xllcorner      " << xo << "\n"
          << "yllcorner      " << yo << "\n"
          << "cellsize       " << dx << "\n"
          << "NODATA_value   " << missing << "\n";

      double x, y, z;
      for (size_t i = ny; i > 0; --i)
      {
        y = yo + (i - 1) * dy;

        for (size_t j = 0; j < nx; ++j)
        {
          x = xo + j * dx;
          z = 1 + x * y;
          fid << z << " ";
        }

        fid << "\n";
      }

      fid.close();
    }

    icepack::GridData example_data = icepack::readArcAsciiGrid(filename);

    const double xmin = example_data.xrange[0];
    const double xmax = example_data.xrange[1];
    const double ymin = example_data.yrange[0];
    const double ymax = example_data.yrange[1];


    CHECK_REAL(xmin, xo, 1.0e-16);
    CHECK_REAL(ymin, yo, 1.0e-16);
    CHECK_REAL(xmax, xo + (nx - 1) * dx, 1.0e-16);
    CHECK_REAL(ymax, yo + (ny - 1) * dy, 1.0e-16);

    double max_diff = 0.0;
    for (size_t i = ny; i > 0; --i)
    {
      const double y = yo + (i - 1) * dy;
      for (size_t j = 0; j < nx; ++j)
      {
        const double x = xo + j * dx;
        const dealii::Point<2> p{x, y};

        const double z = 1 + x * y;
        const double w = example_data.value(p, 0);

        max_diff = std::max(max_diff, std::abs(w - z));
      }
    }

    CHECK_REAL(max_diff, 0.0, 1.0e-15);
  }

  return 0;
}
