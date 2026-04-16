#include <iostream>
#include <vector>

#include "cross_section_reader.h"

int main() {

  const std::string cross_section_dir =
      "/home/ebny-walid-ahammed/github/cross_sections/endfb-vii.1-hdf5/neutron";

  neuxs::OpenMCCrossSectionReader reader(cross_section_dir);

  auto energy = reader.getEnergyDataPoints("U236", 250);
  auto elastic_scattering_cross_section = reader.getCrossSectionDataPoints(
      "U236", 250, neuxs::CrossSectionDataType::SCATTERING);

  if (energy.size() == elastic_scattering_cross_section.size()) {
    std::cout << "We are so cool\n";
  }
  return 0;
}