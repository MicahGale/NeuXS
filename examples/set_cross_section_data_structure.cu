#include <iostream>
#include <vector>

#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "material.cuh"

int main() {

  neuxs::OpenMCCrossSectionReader reader;

  const char *name = "U235"; // C++ will give warning for a weird reason!
  // not anymore lol. made it const char*
  neuxs::NuclideComponent<double> u235(name,
                                       4.8e22f, // atom density (example value)
                                       250.0f,  // temperature in K
                                       true     // fission allowed
  );
  auto aos_data = neuxs::AoSLinear<double>();
  aos_data.setCrossSection(reader, u235);
  auto soa_data = neuxs::SoALinear<double>();
  soa_data.setCrossSection(reader, u235);

  // host-side array access (renamed from _device_data -> _xs_data for clarity)
  auto grids_aos = aos_data._xs_data;

  double soa_sigma_s = soa_data._xs_data._sigma_s[0];
  double soa_sigma_f = soa_data._xs_data._sigma_f[0];
  double soa_sigma_c = soa_data._xs_data._sigma_c[0];
  double soa_sigma_t = soa_data._xs_data._sigma_t[0];

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

  auto aos_view = aos_data.uploadToDevice();
  auto soa_view = soa_data.uploadToDevice();

  printf("=================== Device views ==================\n");
  std::cout << "aos_view: size= " << aos_view._size
            << " energy= " << aos_view._energy
            << " sigma_s= " << aos_view._grid;
  std::cout << "\nsoa_view: size= " << soa_view._size
            << " energy= " << soa_view._energy
            << " sigma_s= " << soa_view._data._sigma_s << std::endl;

  return 0;
}