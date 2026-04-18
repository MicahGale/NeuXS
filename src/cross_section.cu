#include <algorithm>
#include <filesystem>

#include "cross_section.cuh"
#include "cross_section_reader.h"

namespace neuxs {

template <typename T>
void AoSLinear<T>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent &nuclide) {

  auto nuclide_name = std::string(nuclide._name);
  this->_energy =
      reader.getEnergyDataPoints<T>(nuclide_name, nuclide._temperature);
  const auto size = this->_energy.size();
  this->_device_data.reserve(size);

  auto scattering = reader.getCrossSectionDataPoints<T>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::SCATTERING);

  auto capture = reader.getCrossSectionDataPoints<T>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::CAPTURE);

  std::vector<T> fission;
  if (nuclide._allows_fission) {
    fission = reader.getCrossSectionDataPoints<T>(
        nuclide_name, nuclide._temperature, CrossSectionDataType::FISSION);
    if (fission.empty())
      fission.reserve(size, 0);

  } else
    fission.reserve(size, 0);

  for (std::size_t i = 0; i < size; i++) {
    CrossSectionGridPoint<T> grid_point(scattering[i], fission[i], capture[i]);
    _device_data[i] = grid_point;
  }
}

template <typename T>
void SoALinear<T>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent &nuclideComponent) {}

}; // namespace neuxs
