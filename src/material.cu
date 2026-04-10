#include "cross_section.cuh"
#include "material.cuh"

namespace neuxs {
template <typename Derived> class Material {
public:
  __device__ void get_total_cross_section(float *energies, float *sigma_t,
                                          unsigned int n_particles) {
    static_cast<Derived>(this)->get_sigma_t(energies, sigma_t, n_particles);
  };

  class ArrayStructMat {
  private:
    CrossSectionGridPoint[] data;
    unsigned int n_points;

  public:
    __device__ void get_sigma_t(float *energies, float *sigma_t,
                                unsigned int n_particles) {
      float energy = energies[threadIdx.x];
      for (unsigned int i = 0; i < this->n_points; i++) {
        if (this->data[i].energy >= energy) {
          CrossSectionGridPoint lower, upper;
          lower = this->data[i - 1];
          upper = this->data[i];
          // TODO interpolate
        }
      }
    }
  };
} // namespace neuxs
