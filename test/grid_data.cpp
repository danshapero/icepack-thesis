
#include <icepack/grid_data.hpp>
#include "testing.hpp"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <sstream>

int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("--verbose");

  TEST_SUITE("manually creating gridded data sets")
  {
    // Make the coordinate values
    const size_t nx = 128, ny = 192;
    const double dx = 2.0, dy = 2.0;
    const double x0 = 3.0, y0 = 7.0;

    const dealii::Point<2> lower(x0, y0);
    const dealii::Point<2> upper(x0 + nx * dx, y0 + ny * dy);

    // Make some synthetic data
    dealii::Table<2, double> data(nx + 1, ny + 1);

    for (size_t i = 0; i <= ny; ++i)
      for (size_t j = 0; j <= nx; ++j)
        data[j][i] = (x0 + j * dx) + (y0 + i * dy);

    // Make one of the measurements a missing data point
    const double missing = -9999.0;
    const size_t I = ny / 2, J = nx / 2;
    data[J][I] = missing;

    dealii::Table<2, bool> mask(nx + 1, ny + 1);
    for (size_t i = 0; i < ny; ++i)
      for (size_t j = 0; j < nx; ++j)
        mask(j, i) = false;
    mask(J, I) = true;

    // Make the gridded data object
    const icepack::GridData<2> grid_data(lower, upper, data, mask);

    const dealii::Point<2> cell_size = grid_data.cell_size();
    CHECK_REAL(cell_size[0], dx, 1.0e-15);
    CHECK_REAL(cell_size[1], dy, 1.0e-15);

    for (size_t i = 0; i <= nx; ++i)
      for (size_t j = 0; j <= ny; ++j)
      {
        const std::array<size_t, 2> index{{i, j}};
        const dealii::Point<2> xij = grid_data.grid_point(index);
        const dealii::Point<2> xij_exact{x0 + i * dx, y0 + j * dy};
        CHECK_REAL(xij.distance(xij_exact), 0.0, 1.0e-15);
      }

    // Check that a point near the missing data is masked
    const dealii::Point<2> x(x0 + (J + 0.5) * dx, y0 + (I + 0.5) * dy);
    CHECK(grid_data.is_masked(x));
    CHECK(not grid_data(x));

    const dealii::Point<2> y(x0 + dx, y0 + dy);
    CHECK(not grid_data.is_masked(y));
    CHECK_REAL(grid_data(y).get(), y[0] + y[1], 1.0e-16);
  }


  TEST_SUITE("reading Arc ASCII grids")
  {
    const size_t nx = 11, ny = 6;
    const double xo = 0.0, yo = 0.0;
    const double dx = 0.2, dy = dx;
    const double missing = -9999.0;

    const std::string filename = "example_arc_grid.txt";

    // Create an Arc ASCII grid file.
    std::ostringstream ostream;
    ostream << "ncols          " << nx << "\n"
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
        ostream << z << " ";
      }

      ostream << "\n";
    }

    const std::string& grid_file = ostream.str();
    std::istringstream istream(grid_file);
    icepack::GridData<2> example_data = icepack::read_arc_ascii_grid(istream);

    const dealii::Point<2> cell_size = example_data.cell_size();
    CHECK_REAL(cell_size[0], dx, 1.0e-15);
    CHECK_REAL(cell_size[1], dx, 1.0e-15);

    double max_diff = 0.0;
    for (size_t i = ny; i > 0; --i)
    {
      const double y = yo + (i - 1) * dy;
      for (size_t j = 0; j < nx; ++j)
      {
        const double x = xo + j * dx;
        const dealii::Point<2> p(x, y);

        const double z = 1 + x * y;
        const double w = example_data.value(p, 0);

        max_diff = std::max(max_diff, std::abs(w - z));
      }
    }

    CHECK_REAL(max_diff, 0.0, 1.0e-15);
  }


  TEST_SUITE("writing Arc ASCII grids")
  {
    const size_t nx = 128, ny = 192;
    const double dx = 2.0, dy = 2.0;
    const double x0 = 3.0, y0 = 7.0;

    const dealii::Point<2> lower(x0, y0);
    const dealii::Point<2> upper(x0 + nx * dx, y0 + ny * dy);

    dealii::Table<2, double> data(nx + 1, ny + 1);

    for (size_t i = 0; i <= ny; ++i)
      for (size_t j = 0; j <= nx; ++j)
        data[j][i] = (x0 + j * dx) + (y0 + i * dy);

    const double missing = -9999.0;
    const size_t I = ny / 2, J = nx / 2;
    data[J][I] = missing;

    dealii::Table<2, bool> mask(nx + 1, ny + 1);
    for (size_t i = 0; i < ny; ++i)
      for (size_t j = 0; j < nx; ++j)
        mask(j, i) = false;
    mask(J, I) = true;

    std::ostringstream ostream;
    const icepack::GridData<2> grid_data(lower, upper, data, mask);
    write_arc_ascii_grid(grid_data, -9999.0, ostream);

    std::istringstream istream(ostream.str());
    const icepack::GridData<2> grid_data_in =
      icepack::read_arc_ascii_grid(istream);

    const std::array<size_t, 2> n_subintervals = grid_data_in.n_subintervals();
    CHECK((n_subintervals[0] == nx) and (n_subintervals[1] == ny));

    for (size_t i = 0; i < nx; ++i)
      for (size_t j = 0; j < ny; ++j)
      {
        CHECK(grid_data.mask(i, j) == grid_data_in.mask(i, j));
        CHECK_REAL(grid_data.data(i, j), grid_data_in.data(i, j), 1.0e-8);
      }
  }


  TEST_SUITE("interpolating finite element fields to regular grids")
  {
    const dealii::Point<2> center(0.0, 0.0);
    const double inner_r = 0.5;
    const double outer_r = 1.0;
    dealii::SphericalManifold<2, 2> surface(center);

    dealii::Triangulation<2> tria;
    dealii::GridGenerator::hyper_shell(tria, center, inner_r, outer_r, 4);
    tria.set_all_manifold_ids(0);
    tria.set_manifold(0, surface);
    const size_t num_levels = 2;
    tria.refine_global(num_levels);

    const size_t p = 1;
    const auto discretization = icepack::make_discretization(tria, p);

    const icepack::testing::AffineFunction Phi(1, dealii::Point<2>(4, 2));
    const icepack::Field<2> phi = icepack::interpolate(*discretization, Phi);

    const double dx = 0.5 * dealii::GridTools::minimal_cell_diameter(tria);
    const icepack::GridData<2> Psi = icepack::field_to_grid(phi, dx);

    const size_t nx = Psi.data.size(0);
    const size_t ny = Psi.data.size(1);

    size_t num_entries = 0;
    for (size_t i = 0; i < nx; ++i)
      for (size_t j = 0; j < ny; ++j)
        num_entries += !Psi.mask(i, j);
    const dealii::Point<2> cell_size = Psi.cell_size();

    if (verbose)
      std::cout << "Grid size: " << nx << ", " << ny << "\n"
                << "Number of valid entries: " << num_entries << "\n"
                << "Resolution: " << cell_size[0] << ", " << cell_size[1]
                << "\n";

    CHECK(num_entries != 0);

    CHECK_REAL(cell_size[0], dx, 1.0e-8);
    CHECK_REAL(cell_size[1], dx, 1.0e-8);

    for (size_t i = 0; i < nx; ++i)
      for (size_t j = 0; j < ny; ++j)
        if (!Psi.mask(i, j))
        {
          const std::array<size_t, 2> index{{i, j}};
          const dealii::Point<2> x = Psi.grid_point(index);
          CHECK_REAL(Psi.data(i, j), Phi.value(x), 1.0e-8);
        }
  }

  return 0;
}

