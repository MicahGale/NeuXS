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

  const std::vector<neuxs::NuclideComponent<FPrecision>> fuel_isotopes = {
      {"U235", 235, 1.15e-6, temperature, true},
      {"U238", 238, 5.45e-5, temperature, true},
      {"O16", 16, 9.79e-4, temperature, false}};

  const std::vector<neuxs::NuclideComponent<FPrecision>> clad_isotopes = {
      {"Zr90", 90, 2.84e-4, temperature, false},
      {"Sn120", 120, 2.26e-6, temperature, false},
      {"Fe56", 56, 2.15e-6, temperature, false},
      {"Cr52", 52, 1.16e-6, temperature, false}};

  const std::vector<neuxs::NuclideComponent<FPrecision>> mod_isotopes = {
      {"H1", 1, 2.99e-2, temperature, false},
      {"O16", 16, 9.34e-4, temperature, false}};
};

template <typename XSDataViewType, typename XSDataStruct, typename FPrecision>
int run_simulation(int number_of_particles, int threads_per_block) {
  using Isotope = neuxs::NuclideComponent<FPrecision>;
  using Cell = neuxs::Cell<XSDataStruct, FPrecision>;
  using Material = neuxs::Material<XSDataStruct, FPrecision>;
  using Particle = neuxs::Particle<FPrecision>;
  using CellViewType = neuxs::CellView<XSDataViewType, FPrecision>;

  neuxs::MemoryManager memory_manager;
  neuxs::StopWatch gpu_timer;

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

  make_material(&fuel_material, pincell.fuel_isotopes);
  make_material(&clad_material, pincell.clad_isotopes);
  make_material(&mod_material, pincell.mod_isotopes);

  Cell fuel_cell(pincell.volume_fuel, 0);
  Cell cladding_cell(pincell.volume_clad, 1);
  Cell moderator_cell(pincell.volume_mod, 2);

  fuel_cell.setMaterial(&fuel_material);
  cladding_cell.setMaterial(&clad_material);
  moderator_cell.setMaterial(&mod_material);

  fuel_cell.setNeighboringCells({&cladding_cell});
  cladding_cell.setNeighboringCells({&moderator_cell, &fuel_cell});
  moderator_cell.setNeighboringCells({&cladding_cell});

  auto device_fuel_cell = fuel_cell.uploadToDevice();
  auto device_cladding_cell = cladding_cell.uploadToDevice();
  auto device_moderator_cell = moderator_cell.uploadToDevice();

  Particle *host_particles = neuxs::get_mono_energetic_particles<FPrecision>(
      number_of_particles, fuel_cell._id);

  Particle *device_particles =
      memory_manager.allocateDevice<Particle>(number_of_particles);
  memory_manager.copyToDevice(host_particles, device_particles,
                              number_of_particles);

  const size_t n_cells = 4;
  int threads = 256;
  int blocks = (number_of_particles + threads - 1) / threads;
  CellViewType *h_cell_ptrs[n_cells] = {
      device_fuel_cell,     // id = 0
      device_cladding_cell, // id = 1
      device_moderator_cell // id = 2
  };

  CellViewType **d_cell_ptrs =
      memory_manager.allocateDevice<CellViewType *>(n_cells);
  memory_manager.copyToDevice(h_cell_ptrs, d_cell_ptrs, n_cells);
  gpu_timer.startClock();
  neuxs::transport_particles<<<blocks, threads>>>(
      device_particles, number_of_particles, d_cell_ptrs, n_cells);
  auto time_elapsed = gpu_timer.stopClock();

  printf("Time taken = %f milli second\n", time_elapsed);

  return 0;

  // I will worry about the cleanup later
}

template <typename FPrecision>
int dispatch_xs(std::string_view xs_type, int n_particles,
                int threads_per_block) {
  if (xs_type == "aos") {
    std::cout << "Using AoSLinear\n";
    return run_simulation<neuxs::AoSLinearView<FPrecision>,
                          neuxs::AoSLinear<FPrecision>, FPrecision>(
        n_particles, threads_per_block);
  }
  if (xs_type == "soa") {
    std::cout << "Using SoALinear\n";
    return run_simulation<neuxs::SoALinearView<FPrecision>,
                          neuxs::SoALinear<FPrecision>, FPrecision>(
        n_particles, threads_per_block);
  }
  if (xs_type == "log") {
    std::cout << "Using LogarithmicHashAoS\n";
    return run_simulation<neuxs::LogarithmicHashAoSView<FPrecision>,
                          neuxs::LogarithmicHashAoS<FPrecision>, FPrecision>(
        n_particles, threads_per_block);
  }
  if (xs_type == "slbw") {
    std::cout << "Using SLBW\n";
    return run_simulation<neuxs::PiecewiseSlbwModelView<FPrecision>,
                          neuxs::PiecewiseSlbwModel<FPrecision>, FPrecision>(n_particles, threads_per_block);
  }
  std::cerr << "Invalid XS type: " << xs_type
            << " (expected aos|soa|log|slbw)\n";
  return 1;
}

} // namespace pincell

#endif // NEUXS_REFLECTIVE_PINCELL_CUH
