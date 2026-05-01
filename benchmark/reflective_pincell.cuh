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

#include <iostream>
#include <string_view>
#include <vector>

#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "geometry.cuh"
#include "material.cuh"
#include "timer.cuh"
#include "transport.cuh"

namespace pincell {

template <typename FPrecision> struct PinCell {
  const FPrecision temperature = 250.0;
  const FPrecision volume_fuel = 67.92;
  const FPrecision volume_mod = 80.306;
  const FPrecision volume_clad = 8.4;
  const FPrecision volume_gas = 1.374;

  const std::vector<neuxs::NuclideComponent<FPrecision>> fuel_isotopes = {
      {"U235", 235, 4.5e-4f, temperature, true},
      {"U238", 238, 2.15e-2f, temperature, true},
      {"O16", 16, 2.60e-2f, temperature, false}};

  const std::vector<neuxs::NuclideComponent<FPrecision>> gas_isotopes = {
      {"C12", 12, 1.0e-6f, temperature, false}};

  const std::vector<neuxs::NuclideComponent<FPrecision>> clad_isotopes = {
      {"Zr90", 90, 4.25e-2f, temperature, false},
      {"Sn120", 120, 4.5e-4f, temperature, false},
      {"Fe56", 56, 2.0e-4f, temperature, false},
      {"Cr52", 52, 1.0e-4f, temperature, false}};

  const std::vector<neuxs::NuclideComponent<FPrecision>> mod_isotopes = {
      {"H1", 1, 4.96e-2f, temperature, false},
      {"O16", 16, 2.48e-2f, temperature, false}};
};

template <typename XSDataStruct, typename FPrecision> int run_simulation() {
  using Isotope = neuxs::NuclideComponent<FPrecision>;
  using Cell = neuxs::Cell<XSDataStruct, FPrecision>;
  using Material = neuxs::Material<XSDataStruct, FPrecision>;

  neuxs::OpenMCCrossSectionReader reader;
  pincell::PinCell<FPrecision> pincell;

  auto make_material = [](Material *material,
                          const std::vector<Isotope> &isotopes) {
    for (const auto &iso : isotopes)
      material->addIsotope(iso);
  };

  Material fuel_material(reader, pincell.fuel_isotopes.size());
  Material clad_material(reader, pincell.clad_isotopes.size());
  Material mod_material(reader, pincell.mod_isotopes.size());
  Material gas_material(reader, pincell.gas_isotopes.size());

  make_material(&fuel_material, pincell.fuel_isotopes);
  make_material(&gas_material, pincell.gas_isotopes);
  make_material(&clad_material, pincell.clad_isotopes);
  make_material(&mod_material, pincell.mod_isotopes);

  Cell fuel_cell(pincell.volume_fuel, 1 /* cell_id*/);
  Cell gas_gap_cell(pincell.volume_gas, 2);
  Cell cladding_cell(pincell.volume_clad, 3);
  Cell moderator_cell(pincell.volume_mod, 4);

  fuel_cell.setMaterial(&fuel_material);
  gas_gap_cell.setMaterial(&gas_material);
  cladding_cell.setMaterial(&clad_material);
  moderator_cell.setMaterial(&mod_material);

  fuel_cell.setNeighboringCells({&gas_gap_cell});
  gas_gap_cell.setNeighboringCells({&fuel_cell, &cladding_cell});
  cladding_cell.setNeighboringCells({&moderator_cell, &gas_gap_cell});
  moderator_cell.setNeighboringCells({&cladding_cell});

  return 0;
}

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