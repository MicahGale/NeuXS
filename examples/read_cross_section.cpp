#include <iostream>
#include <vector>

#include "cross_section.cuh"

int main() {

    const std::string cross_section_dir = "/home/ebny-walid-ahammed/github/cross_sections/endfb-vii.1-hdf5/neutron";

    neuxs::OpenMCCrossSectionReader reader(cross_section_dir);

    auto energy = reader.getEnergyDataPoints("U235", 250);
    auto cross_section = reader.getCrossSectionDataPoints("U235",250, neuxs::CrossSectionDataType::SCATTERING, "elastic");

    if (energy.size()==cross_section.size()) {
        std::cout << "We are so cool\n";
        return 0;
    }
    return 1;
}