#ifndef NEUXS_CROSS_SECTION_CUH
#define NEUXS_CROSS_SECTION_CUH

#include <cuda_runtime.h>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "openmc/cross_sections.h"
#include "openmc/reaction.h"
#include "hdf5.h"

#include "material.cuh"



namespace neuxs{


    enum class CrossSectionDataType {
        ENERGY,
        SCATTERING,
        ABSORPTION,
        FISSION,
        CAPTURE,
        TOTAL
    };

    /*A wrapper class around for reading HDF5 cross-section data
     * mostly using hfd5 and openmc api */
    class OpenMCCrossSectionReader {
    public:
        explicit OpenMCCrossSectionReader(std::string cross_section_dir);

        thrust::host_vector<float> getEnergyDataPoints( const std::string& isotope_name, float temperature);
        thrust::host_vector<float> getCrossSectionDataPoints( const std::string& isotope_name, float temperature, CrossSectionDataType data_type, const std::string& reaction_name);

    private:
        // Do I need to set the return type to host vector as well?
        // I need to ask Micah what he thinks.
        std::vector<float> readDataPointsFromFile( const std::string& isotope_name, const std::string& reaction_name, float temperature, CrossSectionDataType data_type);
        std::string buildFilePath(const std::string& isotope_name) const;
        std::string buildDatasetPath(float temperature, CrossSectionDataType data_type, int mt_number) const;
        void validateInputs(const std::string& isotope_name, float temperature) const;

        const std::string cross_section_dir_;
        static constexpr std::string_view CROSS_SECTION_TYPE = "neutron";
    };




    struct CrossSectionGridPoint{

        CrossSectionGridPoint(float energy, float _sigma_s, float sigma_f,
                              float sigma_t, float sigma_g );
        float _energy;
        float _sigma_s;
        float _sigma_f;
        float _sigma_t;
        float _sigma_g;

    };

    /*array of struct.
     * A future todo note for us: We will come back and
     * implement struct of array data structure as well
     */
    struct NuclideCrossSection{

        unsigned int _material_id;
        thrust::device_vector<CrossSectionGridPoint> _cross_section_grids;
    };

    __device__ void  binary_search(float* particle_energy,unsigned int material_id, CrossSectionDataType reaction_type, float * cross_section);

    __host__ void  build_nuclide_grid(/*some argument but I am just a place-holder for now*/ );

    __device__ CrossSectionGridPoint interpolate(CrossSectionGridPoint* grid_point_a, const CrossSectionGridPoint* grid_point_b);

    __host__ void  transfer_cross_section_data_to_device(unsigned int nuclide_index);


}

#endif //NEUXS_CROSS_SECTION_CUH
