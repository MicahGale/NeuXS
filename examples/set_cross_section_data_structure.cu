#include <iostream>
#include <vector>

#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "material.cuh"

int main() {

  neuxs::OpenMCCrossSectionReader reader;

  const char *name = "U235"; // C++ will give warning for a weird reason!
  neuxs::NuclideComponent<double> u235(name,
                                       4.8e22f, // atom density (example value)
                                       250.0f,  // temperature in K
                                       true     // fission allowed
  );
  auto aos_data = neuxs::AoSLinear<double>();
  aos_data.setCrossSection(reader, u235);
  auto soa_data = neuxs::SoALinear<double>();
  soa_data.setCrossSection(reader, u235);

  // copying the first grid and printing it out
  auto grids_aos = aos_data._device_data;

  double soa_sigma_s = soa_data._device_data._sigma_s[0];
  double soa_sigma_f = soa_data._device_data._sigma_f[0];
  double soa_sigma_c = soa_data._device_data._sigma_c[0];
  double soa_sigma_t = soa_data._device_data._sigma_t[0];

  printf("=================== AoS ==========================\n");
  std::cout << "aos_sigma_s = " << grids_aos[0]._sigma_s << std::endl;
  std::cout << "aos_sigma_f = " << grids_aos[0]._sigma_f << std::endl;
  std::cout << "aos_sigma_c = " << grids_aos[0]._sigma_c << std::endl;
  std::cout << "aos_sigma_t = " << grids_aos[0]._sigma_t << std::endl;

  printf("=================== SoA ==========================\n");
  std::cout << "soa_sigma_s = " << soa_sigma_s << std::endl;
  std::cout << "soa_sigma_f = " << soa_sigma_f << std::endl;
  std::cout << "soa_sigma_c = " << soa_sigma_c << std::endl;
  std::cout << "soa_sigma_t = " << soa_sigma_t << std::endl;
  return 0;
}