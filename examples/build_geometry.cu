#include "geometry.cuh"
#include <cuda_runtime.h>

int main() {

  const float vol = 0.12;
  neuxs::Cell fuel_cell(vol, 1);
  neuxs::Cell gas_gap(vol, 2);
  neuxs::Cell clad_cell(vol, 3);
  neuxs::Cell mod_cell(vol, 1);

  neuxs::Cell gas_neighbors[] = {fuel_cell, clad_cell};
  neuxs::Cell clad_neighbors[] = {gas_gap, mod_cell};
  neuxs::Cell mod_neighbors[] = {fuel_cell, clad_cell};

  fuel_cell.setNeighboringCells(&gas_gap, /*number_of_neighbors*/ 1);
  gas_gap.setNeighboringCells(gas_neighbors, /*number_of_neighbors*/ 2);
  mod_cell.setNeighboringCells(mod_neighbors, /*number_of_neighbors*/ 1);
  clad_cell.setNeighboringCells(clad_neighbors, /*number_of_neighbors*/ 1);

  return 0;
}
