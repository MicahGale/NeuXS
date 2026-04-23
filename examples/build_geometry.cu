#include <cuda_runtime.h>

#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "geometry.cuh"
#include "material.cuh"
#include "memory.cuh"

int main() {

  neuxs::OpenMCCrossSectionReader reader;
  neuxs::MemoryManager memory_manager;
  const char *name = "U235";
  neuxs::NuclideComponent<double> u235(name, 4.8e22f, 250.0f, true);

  return 0;
}