#include <stdexcept>

#include "geometry.cuh"


namespace neuxs {

    Cell::Cell(float volume, unsigned int id)
            : _volume(volume), _id(id), _material(nullptr),
              _neighbor_cells(nullptr), _num_neighbors(0) {}

    void Cell::setMaterial(Material* material) {
        _material = material;
    }

    void Cell::setNeighboringCells(Cell* cells, unsigned int number_of_neighbors) {
        _neighbor_cells = cells;
        _num_neighbors = number_of_neighbors;
        this->checkNeighboringCellIDs();
    }

    __host__ void Cell::checkNeighboringCellIDs(){
        for (unsigned int i = 0; i< _num_neighbors;++i){
            if (_neighbor_cells[i]._id == this->_id)
               throw std::runtime_error("Cell id = "+ std::to_string(_neighbor_cells[i]._id) +
               " has a neighbor with a same ID. A cell can't be it's neighbor!" );
        }
    }

    bool __device__ particleEscapesTheCell(float particle_energy){

        //another placeholder. Will fix this one the material class is implemented
        return true;
    }

    __device__ Cell* Cell::getRandomNeighborCell(float particle_energy) {
        if (_neighbor_cells == nullptr || _num_neighbors == 0) {
            return nullptr;
        }

        // this is just a space holder for now. I will come back
        // and re implement this once the material struct is implemented
        // we will call cross-section look up method here to calculate the
        // which neighbor cells the particle escapes to


        //something like this
        /*
         * auto sigma_t = _material->getTotalCrossSection(particle_energy);
         * auto escape_prob = 1 - exp(- sigma_t * some_value_proportional_to_volume )
         * in that way higher volume cell would have lesser escape probability
         * */

        float* escape_probability_to_some_cell = {}; //call some function to calculate cell escape prob

        float sum = 0.0f;
        for (unsigned int i = 0; i < _num_neighbors; i++) {
            escape_probability_to_some_cell[i] = 1.0f;
            // need to fix that latter
            // once we have a material class
            sum += escape_probability_to_some_cell[i];
        }

        for (unsigned int i = 0; i < _num_neighbors; i++) {
            escape_probability_to_some_cell[i] /= sum;
        }

        float random_number = 0; // will fix that later once we have a random number generator

        float cumulative = 0.0f;
        for (unsigned int i = 0; i < _num_neighbors; i++) {
            cumulative += escape_probability_to_some_cell[i];
            if (random_number <= cumulative) {
                return &_neighbor_cells[i];
            }
        }
    }

    const Material* Cell::getMaterial() const {
        return _material;
    }

}