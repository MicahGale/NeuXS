#include <iostream>
#include <vector>

#include "cross_section_reader.h"
#include "cross_section.cuh"
#include "material.cuh"

int main() {

    neuxs::OpenMCCrossSectionReader reader;

    char* name = "U235";
    neuxs::NuclideComponent u235(
            name,
            92235,
            4.8e22f,     // atom density (example value)
            250.0f,      // temperature in K
            true         // fission allowed
    );
    auto data = neuxs::AoSLinear<double>();
    data.setCrossSection(reader, u235);
    data.
    return 0;
}