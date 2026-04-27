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

template <typename FPrecision> struct PinCell {
  const FPrecision temperature = 205.f;
  const FPrecision volume_fuel = 67.92;
  const FPrecision volume_mod = 80.306;
  const FPrecision volume_clad = 8.4;
  const FPrecision volume_gas = 1.374;

  const std::vector<const char *> kFuelIsotopes = {
      "U235",
      "U238",
      "O16",
  };
  const std::vector<FPrecision> kFuelDensities = {
      4.5e-4,  // U-235
      2.15e-2, // U-238
      2.60e-2, // O-16
  };
  const std::vector<const char *> kGapIsotopes = {
      "He4",
  };
  const std::vector<FPrecision> kGapDensities = {
      1.0e-6,
  };
  const std::vector<const char *> kCladIsotopes = {"Zr90", "Sn120", "Fe56",
                                                   "Cr52"};

  const std::vector<FPrecision> kCladDensities = {4.25e-2, 4.5e-4, 2.0e-4,
                                                  1.0e-4};
  const std::vector<const char *> kModeratorIsotopes = {
      "H1", "O16",
      /*"B10",
      "B11" keeping these here for cases where we would want to
       simulate with poison in the geometry. Since boron only(mostly) absorbs
       thermal neutron it doesn't really affect the spectrum of the neutron.
       SO neutron cross-section query will almost be the same as it's slowing
      down"
      */
  };
  const std::vector<FPrecision> kModeratorDensities = {
      4.96e-2,
      2.48e-2,
  };
};

// helper method to resolve the templated data types
template <typename XSDataStruct, typename FPrecision> int run_simulation() {

  using Isotope = neuxs::NuclideComponent<FPrecision>;
  using Cell = neuxs::Cell<XSDataStruct, FPrecision>;
  using Material = neuxs::Material<XSDataStruct, FPrecision>;

  neuxs::OpenMCCrossSectionReader reader;
  pincell::PinCell<FPrecision> pincell;

  // man I love lamda functions
  auto make_isotopes = [](const std::vector<const char *> &isotopes_name,
                          const std::vector<FPrecision> &densities,
                          FPrecision temperature) -> std::vector<Isotope> {
    std::vector<Isotope> isotope_vector;
    isotope_vector.reserve(isotopes_name.size());

    for (size_t i = 0; i < isotopes_name.size(); i++) {
      isotope_vector.emplace_back(isotopes_name[i], densities[i], temperature,
                                  true);
    }

    return isotope_vector;
  };

  auto make_material = [](Material *material,
                          std::vector<Isotope> &isotope_vector) {
    for (auto isotope : isotope_vector)
      material->addIsotope(isotope);
  };

  auto fuel_isotopes = make_isotopes(
      pincell.kFuelIsotopes, pincell.kFuelDensities, pincell.temperature);
  auto cladding_isotopes = make_isotopes(
      pincell.kCladIsotopes, pincell.kCladDensities, pincell.temperature);
  auto gas_isotopes = make_isotopes(pincell.kGapIsotopes, pincell.kGapDensities,
                                    pincell.temperature);
  auto mod_isotopes =
      make_isotopes(pincell.kModeratorIsotopes, pincell.kModeratorDensities,
                    pincell.temperature);

  // now that isotopes are done building let's make the materials
  Material fuel_material(reader, fuel_isotopes.size());
  Material gas_material(reader, gas_isotopes.size());
  Material clad_material(reader, cladding_isotopes.size());
  Material mod_material(reader, mod_isotopes.size());

  make_material(&fuel_material, fuel_isotopes);
  make_material(&gas_material, gas_isotopes);
  make_material(&clad_material, cladding_isotopes);
  make_material(&mod_material, mod_isotopes);

  return 0;
}

// helper for running the transport kernel
template <typename FPrecision>
int dispatch_xs(std::string_view xs_type)

{
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
