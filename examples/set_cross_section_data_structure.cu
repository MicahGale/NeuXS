#include <iostream>
#include <vector>

#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "material.cuh"

int main() {

  neuxs::OpenMCCrossSectionReader reader;

  char *name = "U235"; // C++ will give warning for a weird reason!
  neuxs::NuclideComponent u235(name, 92235,
                               4.8e22f, // atom density (example value)
                               250.0f,  // temperature in K
                               true     // fission allowed
  );
  auto data = neuxs::AoSLinear<double>();
  data.setCrossSection(reader, u235);

  // copying the first grid and printing it out
  thrust::host_vector<neuxs::CrossSectionGridPoint<double>> grids =
      data._device_data;

  std::cout << "Sigma_s = " << grids[0]._sigma_s << std::endl;
  std::cout << "Sigma_f = " << grids[0]._sigma_f << std::endl;
  std::cout << "Sigma_c = " << grids[0]._sigma_c << std::endl;
  std::cout << "Sigma_t = " << grids[0]._sigma_t << std::endl;

  return 0;
}