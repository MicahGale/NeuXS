#ifndef NEUXS_REFLECTIVE_PINCELL_CUH
#define NEUXS_REFLECTIVE_PINCELL_CUH

#include <iostream>
#include <string_view>
#include <vector>

#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "geometry.cuh"
#include "material.cuh"
#include "timer.cuh"
#include "transport.cuh"

__global__ void
dummy_transport(neuxs::CellView<neuxs::AoSLinearView<float>, float> *cell,
                neuxs::Particle<float> *particle, bool *escape,
                size_t *next_cell_id) {
  if (particle->isAlive()) {
    *escape = cell->particleEscapesTheCell(particle);
    printf("Particle escaped %d \n", *escape);
  }

  if (*escape == true) {
    printf("getting random cells\n");
    printf("particle old cell was %d\n", particle->getCellID());
    *next_cell_id = cell->getRandomNeighborCellIdx(particle);
    printf("next random cell id = %zu\n", *next_cell_id);
  }
}

namespace pincell {

struct PinCell {
  const float temperature = 250.0;
  // in ideal case this isn't the volume.
  // I am making it small so that particle has a higher chance of escaping
  // which I wanted to demonstrate in this problem
  const float volume_fuel = 0.92;
  const float volume_mod = 8.4;

  const std::vector<neuxs::NuclideComponent<float>> fuel_isotopes = {
      {"U235", 235, 1.15e-6f, temperature, true},
      {"U238", 238, 5.45e-5f, temperature, true},
      {"O16", 16, 9.79e-4f, temperature, false}};

  const std::vector<neuxs::NuclideComponent<float>> mod_isotopes = {
      {"H1", 1, 2.99e-2f, temperature, false},
      {"O16", 16, 9.34e-4f, temperature, false}};
};

int run_simulation() {
  using Isotope = neuxs::NuclideComponent<float>;
  using Cell = neuxs::Cell<neuxs::AoSLinear<float>, float>;
  using Material = neuxs::Material<neuxs::AoSLinear<float>, float>;
  using Particle = neuxs::Particle<float>;

  neuxs::MemoryManager memory_manager;

  neuxs::OpenMCCrossSectionReader reader;
  pincell::PinCell pincell;

  auto make_material = [](Material *material,
                          const std::vector<Isotope> &isotopes) {
    for (const auto &iso : isotopes)
      material->addIsotope(iso);
  };

  Material fuel_material(reader, pincell.fuel_isotopes.size());
  Material mod_material(reader, pincell.mod_isotopes.size());
  make_material(&fuel_material, pincell.fuel_isotopes);
  make_material(&mod_material, pincell.mod_isotopes);

  Cell fuel_cell(pincell.volume_fuel, 0);
  Cell moderator_cell(pincell.volume_mod, 1);

  fuel_cell.setMaterial(&fuel_material);
  moderator_cell.setMaterial(&mod_material);
  fuel_cell.setNeighboringCells({&moderator_cell});
  moderator_cell.setNeighboringCells({&fuel_cell});

  auto device_fuel_cell = fuel_cell.uploadToDevice();
  auto device_moderator_cell = moderator_cell.uploadToDevice();

  Particle particle(1e6, 0);
  size_t host_cell_id = particle.getCellID();

  auto *device_escaped_cell_id = memory_manager.allocateDevice<size_t>(1);
  auto *device_particle = memory_manager.allocateDevice<Particle>(1);
  memory_manager.copyToDevice(&particle, device_particle, 1);
  memory_manager.copyToDevice(&host_cell_id, device_escaped_cell_id, 1);

  auto *escape = memory_manager.allocateDevice<bool>(1);
  bool result = false;
  memory_manager.copyToDevice(&result, escape, 1);
  dummy_transport<<<1, 1>>>(device_fuel_cell, device_particle, escape,
                            device_escaped_cell_id);
  memory_manager.copyToHost(&result, escape, 1);

  std::cout << "particle was in cell " << host_cell_id << "\n";
  memory_manager.copyToHost(device_escaped_cell_id, &host_cell_id, 1);
  host_cell_id != particle.getCellID()
      ? std::cout << "particle escaped to cell " << host_cell_id << std::endl
      : std::cout << "particle didn't escape the cell \n";

  return 0;
}

} // namespace pincell

int main() { pincell::run_simulation(); }

#endif // NEUXS_REFLECTIVE_PINCELL_CUH