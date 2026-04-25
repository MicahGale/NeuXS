#include <cstdio>
#include <cuda_runtime.h>

#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "geometry.cuh"
#include "material.cuh"
#include "memory.cuh"

template <typename XSViewType, typename FP>
__global__ void probe_cell(neuxs::CellView<XSViewType, FP> *cell,
                           FP *out_sigma_s_first, FP *out_sigma_t_first,
                           unsigned int *out_num_isotopes) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto *mat = cell->_material;
  *out_num_isotopes = mat->_num_isotopes;

  // first isotope, first grid point
  auto &xs0 = mat->_xs_views[0];
  auto grid = xs0._grid[0];
  *out_sigma_s_first = grid._sigma_s;
  *out_sigma_t_first = grid._sigma_t;
}

int main() {

  using material = neuxs::Material<neuxs::AoSLinear<double>, double>;
  using isotope = neuxs::NuclideComponent<double>;
  using cell = neuxs::Cell<neuxs::AoSLinear<double>, double>;

  neuxs::OpenMCCrossSectionReader reader;
  neuxs::MemoryManager memory_manager;

  const char *U235_name = "U235";
  const char *U236_name = "U236";
  const char *Hydrogen_name = "H1";
  const char *Oxygen_name = "O16";

  isotope u235(U235_name, 4.8e22, 250.0, true);
  isotope u236(U236_name, 4.8e22, 250.0, true);
  isotope hydrogen(Hydrogen_name, 2, 250., false);
  isotope oxygen(Oxygen_name, 2.0, 250, false);

  material fuel(reader, 2);
  fuel.addIsotope(u235);
  fuel.addIsotope(u236);

  material water(reader, 2);
  water.addIsotope(hydrogen);
  water.addIsotope(oxygen);

  cell fuel_cell(0.4, 1);
  cell mod_cell(20, 2);

  fuel_cell.setMaterial(&fuel);
  fuel_cell.setNeighboringCells(&mod_cell, 1);

  mod_cell.setMaterial(&water);
  mod_cell.setNeighboringCells(&fuel_cell, 1);

  // ====================== Deep-copy onto the GPU ======================
  // One call per top-level object — materials get uploaded transitively.
  auto *d_fuel_cell = fuel_cell.uploadToDevice();
  auto *d_mod_cell = mod_cell.uploadToDevice();

  printf("Uploaded fuel_cell to %p, mod_cell to %p\n", (void *)d_fuel_cell,
         (void *)d_mod_cell);

  // ========= Run a tiny probe kernel that follows the pointers ========
  double *d_sigma_s = memory_manager.allocateDevice<double>(1);
  double *d_sigma_t = memory_manager.allocateDevice<double>(1);
  unsigned int *d_n = memory_manager.allocateDevice<unsigned int>(1);

  probe_cell<<<1, 1>>>(d_fuel_cell, d_sigma_s, d_sigma_t, d_n);
  memory_manager.deviceSynchronize();

  double h_sigma_s = 0, h_sigma_t = 0;
  unsigned int h_n = 0;
  CUDA_CHECK(cudaMemcpy(&h_sigma_s, d_sigma_s, sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_sigma_t, d_sigma_t, sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(&h_n, d_n, sizeof(unsigned int), cudaMemcpyDeviceToHost));

  printf("fuel_cell from GPU: num_isotopes=%u, first sigma_s=%g, sigma_t=%g\n",
         h_n, h_sigma_s, h_sigma_t);

  // Host-side sanity check for comparison (same data, read on CPU)
  printf("fuel_cell on host:  num_isotopes=%u, first sigma_s=%g, sigma_t=%g\n",
         fuel.numIsotopes(), fuel._cross_section_data[0]._xs_data[0]._sigma_s,
         fuel._cross_section_data[0]._xs_data[0]._sigma_t);

  cudaFree(d_sigma_s);
  cudaFree(d_sigma_t);
  cudaFree(d_n);

  // DeviceBuffer members in each object free themselves at scope exit.
  return 0;
}
