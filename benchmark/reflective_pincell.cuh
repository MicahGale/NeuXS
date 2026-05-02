#ifndef NEUXS_REFLECTIVE_PINCELL_CUH
#define NEUXS_REFLECTIVE_PINCELL_CUH

#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "geometry.cuh"
#include "material.cuh"
#include "timer.cuh"
#include "transport.cuh"
#include <iostream>
#include <string_view>
#include <vector>

namespace pincell {

template <typename FPrecision> struct PinCell {
  const FPrecision temperature = 250.0;
  const FPrecision volume_fuel = 67.92;
  const FPrecision volume_mod = 80.306;
  const FPrecision volume_clad = 8.4;
  const FPrecision volume_gas = 1.374;

  const std::vector<neuxs::NuclideComponent<FPrecision>> fuel_isotopes = {
      {"U235", 235, 1.15e-6f, temperature, true},
      {"U238", 238, 5.45e-5f, temperature, true},
      {"O16", 16, 9.79e-4f, temperature, false}};

  const std::vector<neuxs::NuclideComponent<FPrecision>> gas_isotopes = {
      {"C12", 12, 5.02e-8f, temperature, false}};

  const std::vector<neuxs::NuclideComponent<FPrecision>> clad_isotopes = {
      {"Zr90", 90, 2.84e-4f, temperature, false},
      {"Sn120", 120, 2.26e-6f, temperature, false},
      {"Fe56", 56, 2.15e-6f, temperature, false},
      {"Cr52", 52, 1.16e-6f, temperature, false}};

  const std::vector<neuxs::NuclideComponent<FPrecision>> mod_isotopes = {
      {"H1", 1, 2.99e-2f, temperature, false},
      {"O16", 16, 9.34e-4f, temperature, false}};
};

template <typename XSDataViewType, typename XSDataStruct, typename FPrecision>
int run_simulation() {
  using Isotope = neuxs::NuclideComponent<FPrecision>;
  using Cell = neuxs::Cell<XSDataStruct, FPrecision>;
  using Material = neuxs::Material<XSDataStruct, FPrecision>;
  using Particle = neuxs::Particle<FPrecision>;

  neuxs::MemoryManager memory_manager;

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

  Cell fuel_cell(pincell.volume_fuel, 0);
  Cell gas_gap_cell(pincell.volume_gas, 1);
  Cell cladding_cell(pincell.volume_clad, 2);
  Cell moderator_cell(pincell.volume_mod, 3);

  fuel_cell.setMaterial(&fuel_material);
  gas_gap_cell.setMaterial(&gas_material);
  cladding_cell.setMaterial(&clad_material);
  moderator_cell.setMaterial(&mod_material);

  fuel_cell.setNeighboringCells({&gas_gap_cell});
  gas_gap_cell.setNeighboringCells({&fuel_cell, &cladding_cell});
  cladding_cell.setNeighboringCells({&moderator_cell, &gas_gap_cell});
  moderator_cell.setNeighboringCells({&cladding_cell});

  auto device_fuel_cell = fuel_cell.uploadToDevice();
  auto device_gas_gap_fuel_cell = gas_gap_cell.uploadToDevice();
  auto device_cladding_cell = cladding_cell.uploadToDevice();
  auto device_moderator_cell = moderator_cell.uploadToDevice();

  Particle *host_particles =
      neuxs::get_mono_energetic_particles<FPrecision>(512, fuel_cell._id);
  return 0;

  // I will worry about the cleanup later
}

template <typename FPrecision> int dispatch_xs(std::string_view xs_type) {
  if (xs_type == "aos") {
    std::cout << "Using AoSLinear\n";
    return run_simulation<neuxs::AoSLinearView<FPrecision>,
                          neuxs::AoSLinear<FPrecision>, FPrecision>();
  }
  if (xs_type == "soa") {
    std::cout << "Using SoALinear\n";
    return run_simulation<neuxs::SoALinearView<FPrecision>,
                          neuxs::SoALinear<FPrecision>, FPrecision>();
  }
  if (xs_type == "log") {
    std::cout << "Using LogarithmicHashAoS\n";
    return run_simulation<neuxs::LogarithmicHashAoSView<FPrecision>,
                          neuxs::LogarithmicHashAoS<FPrecision>, FPrecision>();
  }
  std::cerr << "Invalid XS type: " << xs_type << " (expected aos|soa|log)\n";
  return 1;
}

} // namespace pincell

#endif // NEUXS_REFLECTIVE_PINCELL_CUH