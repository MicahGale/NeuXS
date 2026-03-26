#include <iostream>
#include <vector>

#include "cross_section.cuh"

int main() {

    const std::string cross_section_dir = "cross_section_files";

    neuxs::OpenMCCrossSectionReader reader(cross_section_dir);

    auto energy = reader.getEnergyDataPoints("U235", 250);
    auto elastic_scattering_cross_section = reader.getCrossSectionDataPoints("U235",250, neuxs::CrossSectionDataType::SCATTERING, "elastic");

    if (energy.size()==elastic_scattering_cross_section.size()) {

        return 0;
    }
    return 1;
}