#include "hdf5.h"
#include <iostream>
#include <vector>

int main() {
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    file_id = H5Fopen("/home/ebny-walid-ahammed/github/cross_sections/endfb-vii.1-hdf5/neutron/U232.h5",
                      H5F_ACC_RDONLY, H5P_DEFAULT);

    if (file_id < 0) {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    dataset_id = H5Dopen(file_id, "/U232/energy/0K", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Failed to open dataset\n";
        H5Fclose(file_id);
        return 1;
    }

    dataspace_id = H5Dget_space(dataset_id);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    std::vector<double> energy(dims[0]);
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, energy.data());

    if (status >= 0) {
        std::cout << "Energy grid at 294K:\n";
        for (size_t i = 0; i < dims[0]; i++) {
            std::cout << energy[i] << " ";
        }
        std::cout << std::endl;
    }

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return 0;
}