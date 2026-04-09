#include "cross_section.cuh"
#include <cassert>
#include <iostream>

int main() {

    neuxs::OpenMCCrossSectionReader reader("/dummy");

    try {
        assert(reader.buildDatasetPath(250.0f, neuxs::CrossSectionDataType::Elastic, neuxs::getMTNumber(neuxs::CrossSectionDataType::Elastic)) == "/reactions/reaction_002/250K/xs");
        assert(reader.buildDatasetPath(300.0f, neuxs::CrossSectionDataType::Fission, neuxs::getMTNumber(neuxs::CrossSectionDataType::Fission)) =="/reactions/reaction_018/300K/xs");
        assert(reader.buildDatasetPath(600.0f, neuxs::CrossSectionDataType::Capture, neuxs::getMTNumber(neuxs::CrossSectionDataType::Capture)) == "/reactions/reaction_102/600K/xs");
        assert(reader.buildDatasetPath(500.0f, neuxs::CrossSectionDataType::Energy, 0) == "/energy/500K");
        assert(reader.buildDatasetPath(700.0f, neuxs::CrossSectionDataType::Elastic, 10) == "/reactions/reaction_010/700K/xs");
        return 0;
    }
    catch (...){
        return 1;
    };

    return 0;
}