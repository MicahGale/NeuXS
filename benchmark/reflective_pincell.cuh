#ifndef NEUXS_REFLECTIVE_PINCELL_CUH
#define NEUXS_REFLECTIVE_PINCELL_CUH

/**
 * pin_cell_main.cu — driver for a single-pincell neutron transport problem.
 *
 * This pin-cell model is used for cross-section lookup and neutron transport
 * in a thermal reactor. Geometry and dimensions are representative of a
 * typical UO2 fuel rod with Zircaloy cladding and light water moderator.
 *
 * ---------------------------
 * Fuel region (UO2)
 * ---------------------------
 * radius_fuel      = 0.93/2 cm
 * height_fuel      = 100.0 cm (1 m)
 * volume_fuel      = 67.92 cm^3
 *
 * ---------------------------
 * Gap region (He)
 * ---------------------------
 * thickness_gap    ≈ 0.01 cm
 * radius_gap_outer = 0.94 cm
 * volume_gap       ≈ 1.374 cm^3
 *
 * ---------------------------
 * Cladding (Zircaloy)
 * ---------------------------
 * radius_clad_inner = 0.94/2 cm
 * radius_clad_outer = 0.997/2 cm
 * thickness_clad    ≈ 0.057/2 cm
 * volume_clad       ≈ 8.4 cm^3
 *
 * ---------------------------
 * Lattice / Moderator (H2O)
 * ---------------------------
 * lattice_pitch     = 1.26 cm (square pitch)
 * cell_height       = 100.0 cm
 * cell_volume       = 158.8 cm^3
 * mod_volume        = 80.306
 *
 * Number densities are in [atoms/(barn·cm)]. (1 barn = 1e-24 cm^2.)
 * Assumes UO2 at 4.0 wt% U-235, 10.4 g/cm^3; Zircaloy-4 at 6.55 g/cm^3;
 * H2O at 0.743 g/cm^3 (hot operating, ~580 K). Doppler / density feedback
 * not applied here.
 *
 * Usage:
 *   ./this_executable(I need to create another cmake for benchmark)
 * <aos|soa|log> <double|single>
 */

#include <array>
#include <iostream>
#include <string_view>

// neuxs includes
#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "geometry.cuh"
#include "material.cuh"
#include "timer.cuh"
#include "transport.cuh"

namespace pincell {

constexpr double volume_fuel = 67.92;
constexpr double volume_mod = 80.306;
constexpr double volume_clad = 8.4;
constexpr double volume_gas = 1.374;

constexpr std::array<const char *, 3> kFuelIsotopes = {
    "U235",
    "U238",
    "O16",
};
constexpr std::array<double, 3> kFuelDensities = {
    4.5e-4,  // U-235
    2.15e-2, // U-238
    2.60e-2, // O-16
};

constexpr std::array<const char *, 1> kGapIsotopes = {
    "He4",
};
constexpr std::array<double, 1> kGapDensities = {
    1.0e-6,
};

// Maybe I don't need this many isotopes
constexpr std::array<const char *, 23> kCladIsotopes = {
    "Zr90",  "Zr91",  "Zr92",  "Zr94",  "Zr96",  "Sn112", "Sn114", "Sn115",
    "Sn116", "Sn117", "Sn118", "Sn119", "Sn120", "Sn122", "Sn124", "Fe54",
    "Fe56",  "Fe57",  "Fe58",  "Cr50",  "Cr52",  "Cr53",  "Cr54",
};

constexpr std::array<const char *, 4> kModeratorIsotopes = {
    "H1", "O16",
    /*"B10",
    "B11" keeping these here for cases where we would want to
     simulate with poison in the geometry. Since boron only(mostly) absorbs
     thermal neutron it doesn't really affect the spectrum of the neutron.
     SO neutron cross-section query will almost be the same as it's slowing
    down"
    */
};
constexpr std::array<double, 4> kModeratorDensities = {
    4.96e-2, // H-1
    2.48e-2, // O-16
};

// helper method to resolve the templated data types
template <typename XSDataStruct, typename FPrecision> int run_simulation();

// helper for running the transport kernel
template <typename FPrecision> int dispatch_xs(std::string_view xs_type) {
  if (xs_type == "aos") {
    std::cout << "Using AoSLinear\n";
    return run_simulation<neuxs::AoSLinear<FPrecision>, FPrecision>();
  }
  if (xs_type == "soa") {
    std::cout << "Using SoALinear\n";
    return run_simulation<neuxs::SoALinear<FPrecision>, FPrecision>();
  }
  if (xs_type == "log") {
    std::cout << "Using LogarithmicHashAoS\n";
    return run_simulation<neuxs::LogarithmicHashAoS<FPrecision>, FPrecision>();
  }
  std::cerr << "Invalid XS type: " << xs_type << " (expected aos|soa|log)\n";
  return 1;
}

} // namespace pincell

#endif // NEUXS_REFLECTIVE_PINCELL_CUH
