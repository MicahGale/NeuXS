#include <stdexcept>

#include "geometry.cuh"
#include "fmtmsg.h"


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

    __device__ Cell* Cell::getRandomNeighborCell(float particle_energy) {
        if (_neighbor_cells == nullptr || _num_neighbors == 0) {
            return nullptr;
        }

        float random_number = 0;

        // this is just a space holder for now. I will come back
        // and re implement this once the material struct is implemented
        // we will call cross-section look up method here to calculate the
        // escape probability

        //something like this
        /*
         * auto sigma_t = _material->getTotalCrossSection(particle_energy);
         * auto escape_prob = 1 - exp(- sigma_t * some_value_proportional_to_volume )
         * in that way higher volumetric cell would have lesser escape probability
         * */

        unsigned int idx = static_cast<unsigned int>(random_number * _num_neighbors);

        if (idx >= _num_neighbors)
            idx = _num_neighbors - 1;

        return &_neighbor_cells[idx];
    }

    const Material* Cell::getMaterial() const {
        return _material;
    }

}