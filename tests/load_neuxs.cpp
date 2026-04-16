#include "cross_section.cuh"
#include <cassert>
#include <iostream>

void test_cross_section_data() {
    neuxs::OpenMCCrossSectionReader reader;

    auto energy = reader.getEnergyDataPoints("U236", 250);
    auto xs = reader.getCrossSectionDataPoints(
            "U236",
            250,
            neuxs::CrossSectionDataType::SCATTERING
    );

    assert(!energy.empty());
    assert(energy.size() == xs.size());

    std::cout << "test_cross_section_data passed\n";
}

void test_dataset_path() {
    neuxs::OpenMCCrossSectionReader reader("/dummy");

    assert(reader.buildDatasetPath(
            250.0f,
            neuxs::CrossSectionDataType::SCATTERING,
            neuxs::getMTNumber(neuxs::CrossSectionDataType::SCATTERING)
    ) == "/reactions/reaction_002/250K/xs");

    assert(reader.buildDatasetPath(
            300.0f,
            neuxs::CrossSectionDataType::FISSION,
            neuxs::getMTNumber(neuxs::CrossSectionDataType::FISSION)
    ) == "/reactions/reaction_018/300K/xs");

    assert(reader.buildDatasetPath(
            600.0f,
            neuxs::CrossSectionDataType::CAPTURE,
            neuxs::getMTNumber(neuxs::CrossSectionDataType::CAPTURE)
    ) == "/reactions/reaction_102/600K/xs");

    assert(reader.buildDatasetPath(
            500.0f,
            neuxs::CrossSectionDataType::ENERGY,
            0
    ) == "/energy/500K");

    assert(reader.buildDatasetPath(
            700.0f,
            neuxs::CrossSectionDataType::SCATTERING,
            10
    ) == "/reactions/reaction_010/700K/xs");

    std::cout << "test_dataset_path passed\n";
}

int main() {
    test_cross_section_data();
    test_dataset_path();

    std::cout << "All tests passed\n";
    return 0;
}