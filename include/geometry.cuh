#ifndef NEUXS_GEOMETRY_H
#define NEUXS_GEOMETRY_H

#include <cuda_runtime.h>

namespace neuxs {

    struct Material;

    struct Cell {
        float _volume;
        unsigned int _id;
        Material* _material;
        Cell* _neighbor_cells;
        unsigned int _num_neighbors;

        Cell(float volume, unsigned int id);

        void __host__ setMaterial(Material* material);

        void __host__ setNeighboringCells(Cell* cells, unsigned int number_of_neighbors);

        void __host__ checkNeighboringCellIDs();

        __device__ Cell* getRandomNeighborCell(float particle_energy);

        const Material* getMaterial() const;
    };

}

#endif